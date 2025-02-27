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
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.BQSpaceUtils;
import org.elasticsearch.index.codec.vectors.BQVectorUtils;
import org.elasticsearch.index.codec.vectors.es818.OptimizedScalarQuantizer;
import org.elasticsearch.index.codec.vectors.ivf.DefaultPostingListScorer;
import org.elasticsearch.index.codec.vectors.ivf.OptimizePostingListScorer;
import org.elasticsearch.simdvec.VectorScorerFactory;
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

@Fork(value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector" })
@Warmup(iterations = 3, time = 3)
@Measurement(iterations = 5, time = 3)
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
/**
 * Benchmark that compares various implementation of ipByteBinByte.
 * Run with ./gradlew -p benchmarks run --args 'PostingListEncoderBenchmark'
 */
public class PostingListEncoderBenchmark {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    @Param({ "96", "768", "1024" })
    int dims;

    int numVectors = 2000;
    int numQueries = 10;

    ByteBuffersDataOutput defaultOutput = new ByteBuffersDataOutput();
    ByteBuffersDataOutput optimizedOutput = new ByteBuffersDataOutput();

    BinaryVector[] binaryQueries;
    float[] centroid;

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

        OptimizedScalarQuantizer scalarQuantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.COSINE);
        int[] docIds = new int[numVectors];
        float[][] vectors1 = new float[numVectors][dims];
        float[][] vectors2 = new float[numVectors][];
        centroid = new float[dims];
        for (int i = 0; i < numVectors; i++) {
            docIds[i] = i;
            for (int j = 0; j < dims; j++) {
                vectors1[i][j] = ThreadLocalRandom.current().nextFloat();
                centroid[j] += vectors1[i][j];
            }
            VectorUtil.l2normalize(vectors1[i]);
            vectors2[i] = vectors1[i].clone();
        }
        for (int j = 0; j < dims; j++) {
            centroid[j] /= vectors1.length;
        }
        VectorUtil.l2normalize(centroid);

        DefaultPostingListScorer.writePostingList(defaultOutput, dims, scalarQuantizer, docIds, centroid.clone(), vectors1);
        OptimizePostingListScorer.writePostingList(optimizedOutput, dims, scalarQuantizer, docIds, centroid.clone(), vectors2);

        binaryQueries = new BinaryVector[numQueries];
        for (int i = 0; i < numQueries; i++) {
            float[] query = new float[dims];
            for (int j = 0; j < dims; j++) {
                query[j] = ThreadLocalRandom.current().nextFloat();
            }
            VectorUtil.l2normalize(query);
            binaryQueries[i] = quantizeQuery(query, scalarQuantizer, centroid, COSINE);
        }
    }

    @Benchmark
    public void scoreDefault(Blackhole bh) throws IOException {
        for (BinaryVector query : binaryQueries) {
            TopKnnCollector collector = new TopKnnCollector(100, Integer.MAX_VALUE);
            DefaultPostingListScorer.scorePostingList(
                dims,
                COSINE,
                defaultOutput.toDataInput(),
                query.quantized,
                query.corrections,
                collector,
                null,
                centroid
            );
            bh.consume(collector);
        }
    }

    @Benchmark
    public void scoreOptimize(Blackhole bh) throws IOException {
        for (BinaryVector query : binaryQueries) {
            TopKnnCollector collector = new TopKnnCollector(100, Integer.MAX_VALUE);
            OptimizePostingListScorer.scorePostingList(
                dims,
                COSINE,
                optimizedOutput.toDataInput(),
                query.quantized,
                query.corrections,
                collector,
                null,
                centroid
            );
            bh.consume(collector);
        }
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
