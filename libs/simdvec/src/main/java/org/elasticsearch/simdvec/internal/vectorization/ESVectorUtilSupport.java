/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.simdvec.internal.vectorization;

public interface ESVectorUtilSupport {

    short B_QUERY = 4;

    long ipByteBinByte(byte[] q, byte[] d);

    long ipByteBinByteHigh(byte[] q, byte[] d);

    long ipByteBinByteLow(byte[] q, byte[] d);

    long ipByteBinByte(byte[] q, byte[] d, int bitPos);

    int ipByteBit(byte[] q, byte[] d);

    float ipFloatBit(float[] q, byte[] d);
}
