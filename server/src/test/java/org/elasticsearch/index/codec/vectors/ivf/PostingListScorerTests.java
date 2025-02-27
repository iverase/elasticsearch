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
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.store.ByteBuffersDataInput;
import org.apache.lucene.store.ByteBuffersDataOutput;
import org.apache.lucene.util.VectorUtil;
import org.elasticsearch.index.codec.vectors.BQSpaceUtils;
import org.elasticsearch.index.codec.vectors.BQVectorUtils;
import org.elasticsearch.index.codec.vectors.es818.OptimizedScalarQuantizer;
import org.elasticsearch.test.ESTestCase;

import java.io.IOException;

import static org.hamcrest.Matchers.equalTo;

public class PostingListScorerTests extends ESTestCase {

    public void testDuel() throws IOException {
        OptimizedScalarQuantizer quantizer = new OptimizedScalarQuantizer(VectorSimilarityFunction.COSINE);
        int numberDimensions = randomIntBetween(1, 2048);
        int numVectors = randomIntBetween(2, 2048);
        float[][] vectors1 = new float[numVectors][numberDimensions];
        float[][] vectors2 = new float[numVectors][];
        float[] centroid = new float[numberDimensions];
        int[] docIds = new int[numVectors];
        for (int i = 0; i < numVectors; i++) {
            docIds[i] = i;
            for (int j = 0; j < numberDimensions; j++) {
                vectors1[i][j] = randomFloat();
                centroid[j] += vectors1[i][j];
            }
            VectorUtil.l2normalize(vectors1[i]);
            vectors2[i] = vectors1[i].clone();
        }
        for (int j = 0; j < numberDimensions; j++) {
            centroid[j] /= numVectors;
        }
        VectorUtil.l2normalize(centroid);
        ByteBuffersDataOutput defaultOutput = new ByteBuffersDataOutput();
        DefaultPostingListScorer.writePostingList(defaultOutput, numberDimensions, quantizer, docIds, centroid.clone(), vectors1);
        ByteBuffersDataOutput optimizedOutput = new ByteBuffersDataOutput();
        OptimizePostingListScorer.writePostingList(optimizedOutput, numberDimensions, quantizer, docIds, centroid.clone(), vectors2);

        float[] query = new float[numberDimensions];
        for (int j = 0; j < numberDimensions; j++) {
            query[j] = randomFloat();
        }
        VectorUtil.l2normalize(query);
        byte[] initial = new byte[query.length];
        int discretizeDims = BQVectorUtils.discretize(numberDimensions, 64);
        byte[] quantizeQuery = new byte[BQSpaceUtils.B_QUERY * discretizeDims / 8];
        OptimizedScalarQuantizer.QuantizationResult queryCorrections = quantizer.scalarQuantize(query, initial, (byte) 4, centroid);
        BQSpaceUtils.transposeHalfByte(initial, quantizeQuery);
        TopKnnCollector defaultCollector = new TopKnnCollector(100, Integer.MAX_VALUE);
        ByteBuffersDataInput defaultInput = defaultOutput.toDataInput();
        DefaultPostingListScorer.scorePostingList(
            numberDimensions,
            VectorSimilarityFunction.COSINE,
            defaultInput,
            quantizeQuery,
            queryCorrections,
            defaultCollector,
            null,
            centroid
        );
        assertThat(defaultInput.position(), equalTo(defaultOutput.size()));
        TopKnnCollector optimizeCollector = new TopKnnCollector(100, Integer.MAX_VALUE);
        ByteBuffersDataInput optimseInput = optimizedOutput.toDataInput();
        OptimizePostingListScorer.scorePostingList(
            numberDimensions,
            VectorSimilarityFunction.COSINE,
            optimseInput,
            quantizeQuery,
            queryCorrections,
            optimizeCollector,
            null,
            centroid
        );
        assertThat(optimseInput.position(), equalTo(optimizedOutput.size()));
        ScoreDoc[] defaultDocs = defaultCollector.topDocs().scoreDocs;
        ScoreDoc[] optimizeDocs = optimizeCollector.topDocs().scoreDocs;
        assertThat(defaultDocs.length, equalTo(optimizeDocs.length));
        for (int i = 0; i < defaultDocs.length; i++) {
            assertThat(defaultDocs[i].doc, equalTo(optimizeDocs[i].doc));
            assertThat(defaultDocs[i].score, equalTo(optimizeDocs[i].score));
        }
    }
}
