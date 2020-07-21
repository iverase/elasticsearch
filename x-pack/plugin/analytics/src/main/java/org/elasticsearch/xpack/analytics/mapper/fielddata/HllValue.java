/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

package org.elasticsearch.xpack.analytics.mapper.fielddata;


import java.io.IOException;

/**
 * Per-document Hll value. An Hll sketch consist on a fix length
 * array of bytes.
 */
public abstract class HllValue {

    /**
     * Advance this instance to the next value of the rinLens array
     * @return true if there is a next value
     */
    public abstract boolean next() throws IOException;

    /**
     * the current value of the runLen array
     * @return the current value of the runLen array
     */
    public abstract byte value();

    /**
     * Skips over and discards n bytes of data from this HLL value.
     * @param bytes the number of bytes to skip
     */
    public abstract void skip(int bytes);
}
