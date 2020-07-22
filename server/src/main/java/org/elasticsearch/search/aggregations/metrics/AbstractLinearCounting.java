/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.elasticsearch.search.aggregations.metrics;

import org.apache.lucene.util.packed.PackedInts;
import org.elasticsearch.common.util.IntArray;

/**
 * Linear counter, implemented based on pseudo code from
 * http://static.googleusercontent.com/media/research.google.com/fr//pubs/archive/40671.pdf and its appendix
 * https://docs.google.com/document/d/1gyjfMHy43U9OWBXxfaeG-3MjGzejW1dlpyMwEYAAWEI/view?fullscreen
 *
 * Trying to understand what this class does without having read the paper is considered adventurous.
 */
public abstract class AbstractLinearCounting  {

    public static final int MIN_PRECISION = 4;
    public static final int MAX_PRECISION = 18;
    private static final int P2 = 25;

    private final int p;

    public AbstractLinearCounting(int precision) {
        if (precision < MIN_PRECISION) {
            throw new IllegalArgumentException("precision must be >= 4");
        }
        if (precision > MAX_PRECISION) {
            throw new IllegalArgumentException("precision must be <= 18");
        }
        p = precision;
    }

    protected abstract void addEncoded(int encoded);

    protected abstract int size();

    protected abstract IntArray values();

    public int precision() {
        return p;
    }

    public void collect(long hash) {
        final int k = encodeHash(hash, p);
        addEncoded(k);
    }

    public long cardinality() {
        final long m = 1 << P2;
        final long v = m - size();
        return linearCounting(m, v);
    }


    static long linearCounting(long m, long v) {
        return Math.round(m * Math.log((double) m / v));
    }

    static long mask(int bits) {
        return (1L << bits) - 1;
    }

    /**
     * Encode the hash on 32 bits. The encoded hash cannot be equal to <code>0</code>.
     */
    static int encodeHash(long hash, int p) {
        final long e = hash >>> (64 - P2);
        final long encoded;
        if ((e & mask(P2 - p)) == 0) {
            final int runLen = 1 + Math.min(Long.numberOfLeadingZeros(hash << P2), 64 - P2);
            encoded = (e << 7) | (runLen << 1) | 1;
        } else {
            encoded = e << 1;
        }
        assert PackedInts.bitsRequired(encoded) <= 32;
        assert encoded != 0;
        return (int) encoded;
    }
}
