from __future__ import absolute_import, division, print_function
import docker
import logging
import os
import random
from ..container_manager import (
    create_model_container_label, parse_model_container_label,
    ContainerManager, CLIPPER_DOCKER_LABEL, CLIPPER_MODEL_CONTAINER_LABEL,
    CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL,
    CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL, CLIPPER_INTERNAL_RPC_PORT,
    CLIPPER_INTERNAL_MANAGEMENT_PORT)
from ..exceptions import ClipperException
import subprocess32 as subprocess

logger = logging.getLogger(__name__)


class DockerContainerManager(ContainerManager):
    def __init__(self,
                 docker_ip_address="localhost",
                 clipper_query_port=1337,
                 clipper_management_port=1338,
                 clipper_rpc_port=7000,
                 redis_ip=None,
                 redis_port=6379,
                 docker_network="clipper_network",
                 extra_container_kwargs={}):
        """
        Parameters
        ----------
        docker_ip_address : str, optional
            The public hostname or IP address at which the Clipper Docker
            containers can be accessed via their exposed ports. This should almost always
            be "localhost". Only change if you know what you're doing!
        clipper_query_port : int, optional
            The port on which the query frontend should listen for incoming prediction requests.
        clipper_management_port : int, optional
            The port on which the management frontend should expose the management REST API.
        clipper_rpc_port : int, optional
            The port to start the Clipper RPC service on.
        redis_ip : str, optional
            The address of a running Redis cluster. If set to None, Clipper will start
            a Redis container for you.
        redis_port : int, optional
            The Redis port. If ``redis_ip`` is set to None, Clipper will start Redis on this port.
            If ``redis_ip`` is provided, Clipper will connect to Redis on this port.
        docker_network : str, optional
            The docker network to attach the containers to. You can read more about Docker
            networking in the
            `Docker User Guide <https://docs.docker.com/engine/userguide/networking/>`_.
        extra_container_kwargs : dict
            Any additional keyword arguments to pass to the call to
            :py:meth:`docker.client.containers.run`.
        """
        self.public_hostname = docker_ip_address
        self.clipper_query_port = clipper_query_port
        self.clipper_management_port = clipper_management_port
        self.clipper_rpc_port = clipper_rpc_port
        self.redis_ip = redis_ip
        if redis_ip is None:
            self.external_redis = False
        else:
            self.external_redis = True
        self.redis_port = redis_port
        if docker_network is "host":
            raise ClipperException(
                "DockerContainerManager does not support running Clipper on the "
                "\"host\" docker network. Please pick a different network name")
        self.docker_network = docker_network

        self.docker_client = docker.from_env()
        self.extra_container_kwargs = extra_container_kwargs.copy()

        # Merge Clipper-specific labels with any user-provided labels
        if "labels" in self.extra_container_kwargs:
            self.common_labels = self.extra_container_kwargs.pop("labels")
            self.common_labels.update({CLIPPER_DOCKER_LABEL: ""})
        else:
            self.common_labels = {CLIPPER_DOCKER_LABEL: ""}

        container_args = {
            "network": self.docker_network,
            "detach": True,
        }

        self.extra_container_kwargs.update(container_args)

    def start_clipper(self, query_frontend_image, mgmt_frontend_image,
                      cache_size, redis_cpu_str="0", mgmt_cpu_str="1", query_cpu_str="2-11"):
        try:
            self.docker_client.networks.create(
                self.docker_network, check_duplicate=True)
        except docker.errors.APIError as e:
            logger.debug(
                "{nw} network already exists".format(nw=self.docker_network))

        if not self.external_redis:
            logger.info("Starting managed Redis instance in Docker")
            redis_container = self.docker_client.containers.run(
                'redis:alpine',
                "redis-server --port %s" % self.redis_port,
                name="redis-{}".format(
                    random.randint(0, 100000)),  # generate a random name
                ports={'%s/tcp' % self.redis_port: self.redis_port},
                labels=self.common_labels.copy(),
                cpuset_cpus=redis_cpu_str,
                **self.extra_container_kwargs)
            self.redis_ip = redis_container.name

        mgmt_cmd = "--redis_ip={redis_ip} --redis_port={redis_port}".format(
            redis_ip=self.redis_ip, redis_port=self.redis_port)
        mgmt_labels = self.common_labels.copy()
        mgmt_labels[CLIPPER_MGMT_FRONTEND_CONTAINER_LABEL] = ""
        self.docker_client.containers.run(
            mgmt_frontend_image,
            mgmt_cmd,
            name="mgmt_frontend-{}".format(
                random.randint(0, 100000)),  # generate a random name
            ports={
                '%s/tcp' % CLIPPER_INTERNAL_MANAGEMENT_PORT:
                self.clipper_management_port
            },
            labels=mgmt_labels,
            cpuset_cpus=mgmt_cpu_str,
            **self.extra_container_kwargs)
        query_cmd = "--redis_ip={redis_ip} --redis_port={redis_port}".format(
            redis_ip=self.redis_ip,
            redis_port=self.redis_port,
            cache_size=cache_size)
        query_labels = self.common_labels.copy()
        query_labels[CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL] = ""
        self.docker_client.containers.run(
            query_frontend_image,
            query_cmd,
            name="query_frontend-{}".format(
                random.randint(0, 100000)),  # generate a random name
            ports={
                '4455/tcp': 4455,
                '4456/tcp': 4456,
                '1337/tcp': 1337,
                '%s/tcp' % CLIPPER_INTERNAL_RPC_PORT: self.clipper_rpc_port
            },
            labels=query_labels,
            cpuset_cpus=query_cpu_str,
            **self.extra_container_kwargs)
        self.connect()

    def connect(self):
        # No extra connection steps to take on connection
        return

    def deploy_model(self, name, version, input_type, image, num_replicas=1, **kwargs):
        # Parameters
        # ----------
        # image : str
        #     The fully specified Docker imagesitory to deploy. If using a custom
        #     registry, the registry name must be prepended to the image. For example,
        #     "localhost:5000/my_model_name:my_model_version" or
        #     "quay.io/my_namespace/my_model_name:my_model_version"
        self.set_num_replicas(name, version, input_type, image, num_replicas, **kwargs)

    def _get_replicas(self, name, version):
        containers = self.docker_client.containers.list(
            filters={
                "label":
                "{key}={val}".format(
                    key=CLIPPER_MODEL_CONTAINER_LABEL,
                    val=create_model_container_label(name, version))
            })
        return containers

    def get_num_replicas(self, name, version):
        return len(self._get_replicas(name, version))

    def _add_replica(self, name, version, input_type, image, gpu_num=None, cpu_str=None, use_nvidia_docker=False):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_QUERY_FRONTEND_CONTAINER_LABEL})
        if len(containers) < 1:
            logger.warning("No Clipper query frontend found.")
            raise ClipperException(
                "No Clipper query frontend to attach model container to")
        query_frontend_hostname = containers[0].name
        env_vars = {
            "CLIPPER_MODEL_NAME": name,
            "CLIPPER_MODEL_VERSION": version,
            # NOTE: assumes this container being launched on same machine
            # in same docker network as the query frontend
            "CLIPPER_IP": query_frontend_hostname,
            "CLIPPER_INPUT_TYPE": input_type,
        }
        labels = self.common_labels.copy()
        labels[CLIPPER_MODEL_CONTAINER_LABEL] = create_model_container_label(
            name, version)
        if use_nvidia_docker:
            # Even if a GPU-supported model isn't being deployed on a GPU,
            # we may still need to launch it using nvidia-docker because
            # the model framework may still depend on libcuda
            env = os.environ.copy()
            cmd = ["nvidia-docker", "run", "-d",
                   "--network=%s" % self.docker_network]
            if gpu_num:
                logger.info("Starting {name}:{version} on GPU {gpu_num}".format(
                    name=name, version=version, gpu_num=gpu_num))
                env["NV_GPU"] = str(gpu_num)
            else:
                # We're not running on a GPU, so we should mask all available
                # GPU resources
                cmd.append("-e")
                cmd.append("CUDA_VISIBLE_DEVICES=''")
            for k, v in labels.iteritems():
                cmd.append("-l")
                cmd.append("%s=%s" % (k, v))
            for k, v in env_vars.iteritems():
                cmd.append("-e")
                cmd.append("%s=%s" % (k, v))
            if cpu_str:
                cmd.append("--cpuset-cpus=%s" % cpu_str)
            cmd.append(image)
            logger.info("Docker command: \"%s\"" % cmd)
            subprocess.check_call(cmd, env=env)
        else:
            self.docker_client.containers.run(
                image,
                environment=env_vars,
                labels=labels,
                cpuset_cpus=cpu_str,
                **self.extra_container_kwargs)

    def set_num_replicas(self, name, version, input_type, image, num_replicas, **kwargs):
        current_replicas = self._get_replicas(name, version)
        if len(current_replicas) < num_replicas:
            num_missing = num_replicas - len(current_replicas)
            logger.info(
                "Found {cur} replicas for {name}:{version}. Adding {missing}".
                format(
                    cur=len(current_replicas),
                    name=name,
                    version=version,
                    missing=(num_missing)))
            if "gpus" in kwargs:
                available_gpus = list(kwargs["gpus"])
            if "use_nvidia_docker" in kwargs:
                use_nvidia_docker = kwargs["use_nvidia_docker"]
            else:
                use_nvidia_docker = False

            # Enumerated list of cpus that can be allocated (e.g [1, 2, 3, 8, 9])
            if "allocated_cpus" in kwargs:
                allocated_cpus = kwargs["allocated_cpus"]
            if "cpus_per_replica" in kwargs:
                cpus_per_replica = kwargs["cpus_per_replica"]
            if (len(allocated_cpus) / cpus_per_replica) < num_missing:
                raise ClipperException(
                    "Not enough cpus available. Trying to allocate {reps} replicas \
                    {cpus_per} CPUs each out of only {alloc_cpus} allocated cpus".format(
                        reps=num_missing,
                        cpus_per=cpus_per_replica,
                        alloc_cpus=len(allocated_cpus)))
            for i in range(num_missing):
                if len(available_gpus) > 0:
                    gpu_num = available_gpus.pop()
                    use_nvidia_docker = True
                else:
                    gpu_num = None
                cpus = allocated_cpus[i*cpus_per_replica: (i+1)*cpus_per_replica]
                cpus = [str(c) for c in cpus]
                cpu_str = ",".join(cpus)

                self._add_replica(name, version, input_type, image, gpu_num=gpu_num,
                                  cpu_str=cpu_str, use_nvidia_docker=use_nvidia_docker)
        elif len(current_replicas) > num_replicas:
            num_extra = len(current_replicas) - num_replicas
            logger.info(
                "Found {cur} replicas for {name}:{version}. Removing {extra}".
                format(
                    cur=len(current_replicas),
                    name=name,
                    version=version,
                    extra=(num_extra)))
            while len(current_replicas) > num_replicas:
                cur_container = current_replicas.pop()
                cur_container.stop()

    def get_logs(self, logging_dir):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_DOCKER_LABEL})
        logging_dir = os.path.abspath(os.path.expanduser(logging_dir))

        log_files = []
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
            logger.info("Created logging directory: %s" % logging_dir)
        for c in containers:
            log_file_name = "image_{image}:container_{id}.log".format(
                image=c.image.short_id, id=c.short_id)
            log_file = os.path.join(logging_dir, log_file_name)
            with open(log_file, "w") as lf:
                lf.write(c.logs(stdout=True, stderr=True))
            log_files.append(log_file)
        return log_files

    def stop_models(self, models):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_MODEL_CONTAINER_LABEL})
        for c in containers:
            c_name, c_version = parse_model_container_label(
                c.labels[CLIPPER_MODEL_CONTAINER_LABEL])
            if c_name in models and c_version in models[c_name]:
                c.stop()

    def stop_all_model_containers(self):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_MODEL_CONTAINER_LABEL})
        for c in containers:
            c.stop()

    def stop_all(self):
        containers = self.docker_client.containers.list(
            filters={"label": CLIPPER_DOCKER_LABEL})
        for c in containers:
            c.stop()

    def get_admin_addr(self):
        return "{host}:{port}".format(
            host=self.public_hostname, port=self.clipper_management_port)

    def get_query_addr(self):
        return "{host}:{port}".format(
            host=self.public_hostname, port=self.clipper_query_port)
