package com.clipper.container.app.data;

import java.nio.ByteBuffer;

public abstract class DataBuffer<T> {
  abstract void init(ByteBuffer buffer);

  abstract T get(int offset, int size);

  abstract T getAll();
}
