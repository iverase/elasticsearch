/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.ivf;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.DataInput;
import org.apache.lucene.store.DataOutput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.BQVectorUtils;
import org.elasticsearch.index.codec.vectors.es818.OptimizedScalarQuantizer;
import org.elasticsearch.simdvec.ESVectorUtil;

import java.io.IOException;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.index.VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT;

public class OptimizePostingListScorer {

    private static final float FOUR_BIT_SCALE = 1f / ((1 << 4) - 1);
    private static final int BLOCK_SIZE = 256;

    public static void writePostingList(
        DataOutput output,
        int numberDimensions,
        OptimizedScalarQuantizer scalarQuantizer,
        int[] docIds,
        float[] centroid,
        float[][] vectors
    ) throws IOException {
        output.writeVInt(vectors.length);
        int discreteDims = BQVectorUtils.discretize(numberDimensions, 64);
        byte[] quantizationScratch = new byte[discreteDims];
        byte[][] quantizeVectors = new byte[BLOCK_SIZE][discreteDims / 8];
        OptimizedScalarQuantizer.QuantizationResult[] corrections = new OptimizedScalarQuantizer.QuantizationResult[BLOCK_SIZE];
        for (int i = 0; i < vectors.length;) {
            int size = Math.min(BLOCK_SIZE, vectors.length - i);
            // write docIds
            // TODO: can we compress the same way we do it in the BKD tree. Order vectors by docId?
            for (int j = 0; j < size; j++) {
                output.writeInt(docIds[i + j]);
            }
            for (int j = 0; j < size; j++) {
                float[] v = vectors[i + j];
                corrections[j] = scalarQuantizer.scalarQuantize(v, quantizationScratch, (byte) 1, centroid);
                BQVectorUtils.packAsBinary(quantizationScratch, quantizeVectors[j]);
            }
            // write all additionalCorrection (ax)
            for (int j = 0; j < size; j++) {
                output.writeInt(Float.floatToIntBits(corrections[j].additionalCorrection()));
            }
            // write all lowerInterval (ax)
            for (int j = 0; j < size; j++) {
                output.writeInt(Float.floatToIntBits(corrections[j].lowerInterval()));
            }
            // write all (upperInterval - lowerInterval)
            for (int j = 0; j < size; j++) {
                output.writeInt(Float.floatToIntBits(corrections[j].upperInterval() - corrections[j].lowerInterval()));
            }
            // write all quantizedComponentSum
            for (int j = 0; j < size; j++) {
                output.writeShort((short) corrections[j].quantizedComponentSum());
            }
            // write all additional correction and vector
            for (int j = 0; j < size; j++) {
                output.writeBytes(quantizeVectors[j], quantizeVectors[j].length);
            }
            i += size;
        }
    }

    public static boolean scorePostingList(
        int numberDimensions,
        VectorSimilarityFunction similarityFunction,
        DataInput input,
        byte[] query,
        OptimizedScalarQuantizer.QuantizationResult queryCorrections,
        KnnCollector knnCollector,
        Bits acceptDocs,
        float[] centroid
    ) throws IOException {
        byte[] binaryCode = new byte[BQVectorUtils.discretize(numberDimensions, 64) / 8];
        float centroidDp = VectorUtil.dotProduct(centroid, centroid);
        int[] docIds = new int[BLOCK_SIZE];
        float[] scratch1 = new float[BLOCK_SIZE];
        float[] scratch2 = new float[BLOCK_SIZE];
        int vectors = input.readVInt();
        for (int i = 0; i < vectors;) {
            int size = Math.min(BLOCK_SIZE, vectors - i);
            if (scoreBlock(
                numberDimensions,
                similarityFunction,
                input,
                knnCollector,
                acceptDocs,
                size,
                docIds,
                scratch1,
                scratch2,
                query,
                binaryCode,
                queryCorrections,
                centroidDp
            )) {
                return true;
            }
            i += size;
        }
        return false;
    }

    // TODO: too many arguments, consider refactoring
    private static boolean scoreBlock(
        int numberDimensions,
        VectorSimilarityFunction similarityFunction,
        DataInput input,
        KnnCollector knnCollector,
        Bits acceptDocs,
        int blockSize,
        int[] docIds,
        float[] scratch1,
        float[] scratch2,
        byte[] quantizedQuery,
        byte[] binaryCode,
        OptimizedScalarQuantizer.QuantizationResult queryCorrections,
        float centroidDp
    ) throws IOException {
        float ay = queryCorrections.lowerInterval();
        float ly = (queryCorrections.upperInterval() - ay) * FOUR_BIT_SCALE;
        float y1 = queryCorrections.quantizedComponentSum();
        // read docIds
        input.readInts(docIds, 0, blockSize);
        // read additionalCorrection in scratch2
        input.readFloats(scratch2, 0, blockSize);
        // read lower intervals in scratch1
        input.readFloats(scratch1, 0, blockSize);
        // compute ax * ay * dim + ax * ly * y1 +ay * lx * x1 + additionalCorrection in scratch1
        for (int i = 0; i < blockSize; i++) {
            scratch1[i] = scratch2[i] + scratch1[i] * ay * numberDimensions + scratch1[i] * ly * y1;
        }
        // read (upperInterval - lowerInterval)
        input.readFloats(scratch2, 0, blockSize);
        // compute lx * ay * x1 in scratch1
        for (int i = 0; i < blockSize; i++) {
            int quantizedComponentSum = input.readShort() & 0xffff;
            scratch1[i] += scratch2[i] * ay * quantizedComponentSum;
        }
        // compute lx * ly in scratch2
        for (int i = 0; i < blockSize; i++) {
            scratch2[i] *= ly;
        }
        for (int i = 0; i < blockSize; i++) {
            int docId = docIds[i];
            if (acceptDocs != null && acceptDocs.get(docId) == false) {
                input.skipBytes(binaryCode.length + Float.BYTES);
                continue;
            }
            input.readBytes(binaryCode, 0, binaryCode.length);
            float qcDist = ESVectorUtil.ipByteBinByte(quantizedQuery, binaryCode);
            float score = scratch1[i] + scratch2[i] * qcDist;
            knnCollector.incVisitedCount(1);
            knnCollector.collect(
                docId,
                addCorrectionsToScore(similarityFunction, score, queryCorrections.additionalCorrection(), centroidDp)
            );
            if (knnCollector.earlyTerminated()) {
                return true;
            }
        }
        return false;
    }

    private static float addCorrectionsToScore(
        VectorSimilarityFunction similarityFunction,
        float score,
        float queryCorrection,
        float centroidDp
    ) {
        if (similarityFunction == EUCLIDEAN) {
            score = queryCorrection - 2f * score;
            return Math.max(1f / (1f + score), 0);
        } else {
            // For cosine and max inner product, we need to apply the additional correction, which is
            // assumed to be the non-centered dot-product between the vector and the centroid
            score += queryCorrection - centroidDp;
            if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
                return VectorUtil.scaleMaxInnerProductScore(score);
            }
            return Math.max((1f + score) / 2f, 0f);
        }
    }

    private static long removeCorrectionsToScore(
        VectorSimilarityFunction similarityFunction,
        float score,
        OptimizedScalarQuantizer.QuantizationResult queryCorrections,
        float centroidDp
    ) {
        float corrected;
        // For euclidean, we need to invert the score and apply the additional correction, which is
        // assumed to be the squared l2norm of the centroid centered vectors.
        if (similarityFunction == EUCLIDEAN) {
            corrected = score == 0 ? -1 : (1 / score) - 1;
            corrected = (queryCorrections.additionalCorrection() - corrected) * 0.5f;
        } else {
            // For cosine and max inner product, we need to apply the additional correction, which is
            // assumed to be the non-centered dot-product between the vector and the centroid
            if (similarityFunction == MAXIMUM_INNER_PRODUCT) {
                if (score < 1) {
                    corrected = 1 - (1 / score);
                } else {
                    corrected = score - 1;
                }
            } else {
                corrected = score == 0 ? -1 : 2 * score - 1;
            }
            corrected -= queryCorrections.additionalCorrection() - centroidDp;
        }
        return Math.round(corrected);
    }
}
