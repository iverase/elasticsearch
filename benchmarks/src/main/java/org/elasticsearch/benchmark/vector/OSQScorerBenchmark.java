/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.benchmark.vector;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.BQSpaceUtils;
import org.elasticsearch.index.codec.vectors.BQVectorUtils;
import org.elasticsearch.index.codec.vectors.es818.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.ESVectorUtil;
import org.elasticsearch.simdvec.VectorScorerFactory;
import org.elasticsearch.simdvec.internal.vectorization.ESVectorUtilSupport;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

import static org.apache.lucene.index.VectorSimilarityFunction.COSINE;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

@Fork(value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector" })
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
/**
 * Benchmark that compares various implementation of ipByteBinByte.
 * Run with ./gradlew -p benchmarks run --args 'OSQScorerBenchmark'
 */
public class OSQScorerBenchmark {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    private static final float FOUR_BIT_SCALE = 1f / ((1 << 4) - 1);

    @Param({ "1024" })
    int dims;

    OptimizedScalarQuantizer scalarQuantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.COSINE);

    int numVectors = 1000;
    int numQueries = 10;

    float[][] vectors;
    float[][] queries;
    float[] centroid;

    BinaryVector[] binaryVectors;
    BinaryVector[] binaryQueries;

    @Setup
    public void setup() throws IOException {
        var optionalVectorScorerFactory = VectorScorerFactory.instance();
        if (optionalVectorScorerFactory.isEmpty()) {
            String msg = "JDK=["
                + Runtime.version()
                + "], os.name=["
                + System.getProperty("os.name")
                + "], os.arch=["
                + System.getProperty("os.arch")
                + "]";
            throw new AssertionError("Vector scorer factory not present. Cannot run the benchmark. " + msg);
        }

        vectors = new float[numVectors][dims];
        centroid = new float[dims];
        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < dims; j++) {
                vectors[i][j] = ThreadLocalRandom.current().nextFloat();
                centroid[j] += vectors[i][j];
            }
            VectorUtil.l2normalize(vectors[i]);
        }
        for (int j = 0; j < dims; j++) {
            centroid[j] /= vectors.length;
        }
        VectorUtil.l2normalize(centroid);
        binaryVectors = new BinaryVector[numVectors];
        for (int i = 0; i < binaryVectors.length; i++) {
            binaryVectors[i] = quantizeVector(vectors[i], scalarQuantizer, centroid);
        }

        queries = new float[numQueries][dims];
        for (int i = 0; i < numQueries; i++) {
            for (int j = 0; j < dims; j++) {
                queries[i][j] = ThreadLocalRandom.current().nextFloat();
            }
            VectorUtil.l2normalize(queries[i]);
        }
        binaryQueries = new BinaryVector[numQueries];
        for (int i = 0; i < numQueries; i++) {
            binaryQueries[i] = quantizeQuery(queries[i], scalarQuantizer, centroid, COSINE);
        }
    }

    @Benchmark
    public void scoreBinaryVectorUtil(Blackhole bh) {
        score(bh, ESVectorUtil::ipByteBinByte);
    }

    @Benchmark
    public void scoreBinaryLoopVectorUtil(Blackhole bh) {
        score(bh, (q, v) -> {
            long res = 0;
            for (int i = ESVectorUtilSupport.B_QUERY - 1; i >= 0; i--) {
                res += ESVectorUtil.ipByteBinByte(q, v, i);
            }
            return res;
        });
    }

    @Benchmark
    public void scoreBinaryHighLowVectorUtil(Blackhole bh) {
        score(bh, (q, v) -> ESVectorUtil.ipByteBinByteHigh(q, v) + ESVectorUtil.ipByteBinByteLow(q, v));
    }

    private void score(Blackhole bh, VectorScorer scorer) {
        VectorSimilarityFunction similarityFunction = COSINE;
        float centroidDp = VectorUtil.dotProduct(centroid, centroid);
        for (BinaryVector query : binaryQueries) {
            float ay = query.corrections().lowerInterval();
            float ly = (query.corrections().upperInterval() - ay) * FOUR_BIT_SCALE;
            float y1 = query.corrections().quantizedComponentSum();
            for (BinaryVector vector : binaryVectors) {
                float qcDist = scorer.score(query.quantized(), vector.quantized());
                float x1 = vector.corrections().quantizedComponentSum();
                float ax = vector.corrections().lowerInterval();
                // Here we assume `lx` is simply bit vectors, so the scaling isn't necessary
                float lx = vector.corrections().upperInterval() - ax;
                float a = ax * ay * dims + ay * lx * x1 + ay * lx * x1 + ax * ly * y1;
                float b = lx * ly;
                float score = a + b * qcDist;
                // For euclidean, we need to invert the score and apply the additional correction, which is
                // assumed to be the squared l2norm of the centroid centered vectors.
                if (similarityFunction == VectorSimilarityFunction.EUCLIDEAN) {
                    score = query.corrections().additionalCorrection() + vector.corrections().additionalCorrection() - 2 * score;
                    bh.consume(Math.max(1 / (1f + score), 0));
                } else {
                    // For cosine and max inner product, we need to apply the additional correction, which is
                    // assumed to be the non-centered dot-product between the vector and the centroid
                    score += query.corrections().additionalCorrection() + vector.corrections().additionalCorrection() - centroidDp;
                    if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
                        bh.consume(VectorUtil.scaleMaxInnerProductScore(score));
                    } else {
                        bh.consume(Math.max((1f + score) / 2f, 0));
                    }
                }
            }
        }
    }

    private interface VectorScorer {
        float score(byte[] query, byte[] vector);
    }

    private static BinaryVector quantizeVector(float[] vector, OptimizedScalarQuantizer quantizer, float[] centroid) {
        int discretizeDims = BQVectorUtils.discretize(vector.length, 64);
        byte[] initial = new byte[vector.length];
        byte[] quantized = new byte[discretizeDims / 8];
        OptimizedScalarQuantizer.QuantizationResult queryCorrections = quantizer.scalarQuantize(vector, initial, (byte) 1, centroid);
        BQVectorUtils.packAsBinary(initial, quantized);
        return new BinaryVector(quantized, queryCorrections);
    }

    private static BinaryVector quantizeQuery(
        float[] target,
        OptimizedScalarQuantizer quantizer,
        float[] centroid,
        VectorSimilarityFunction similarityFunction
    ) {
        // We make a copy as the quantization process mutates the input
        float[] copy = ArrayUtil.copyOfSubArray(target, 0, target.length);
        if (similarityFunction == COSINE) {
            VectorUtil.l2normalize(copy);
        }
        byte[] initial = new byte[copy.length];
        int discretizeDims = BQVectorUtils.discretize(copy.length, 64);
        byte[] quantized = new byte[BQSpaceUtils.B_QUERY * discretizeDims / 8];
        OptimizedScalarQuantizer.QuantizationResult queryCorrections = quantizer.scalarQuantize(copy, initial, (byte) 4, centroid);
        BQSpaceUtils.transposeHalfByte(initial, quantized);
        return new BinaryVector(quantized, queryCorrections);
    }

    private record BinaryVector(byte[] quantized, OptimizedScalarQuantizer.QuantizationResult corrections) {}

}
