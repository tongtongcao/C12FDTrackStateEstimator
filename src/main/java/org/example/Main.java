package org.example;

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.io.IOException;
import java.nio.file.Paths;

/**
 * Example DJL (Deep Java Library) inference code for a PyTorch Transformer model
 * trained to predict initial track states from a variable-length list of hits.
 *
 * The workflow is:
 *   1. Normalize input hits using training statistics
 *   2. Run inference with a TorchScript model
 *   3. De-normalize the predicted track state
 */
public class Main {

    public static void main(String[] args)
            throws IOException, ModelException, TranslateException {

        // ------------------------------------------------------------
        // Hit normalization parameters (mean / std)
        // Order: (doca, xm, xr, yr, z)
        // These must exactly match the statistics used during training
        float[] hitMean = new float[]{
                0.54767877f, -39.37779f, -39.37546f, 60.905655f, 374.25867f
        };
        float[] hitStd = new float[]{
                0.4348458f, 51.83236f, 52.26446f, 35.65593f, 111.37744f
        };

        // ------------------------------------------------------------
        // Track state de-normalization parameters (mean / std)
        // Order: (x, y, tx, ty, Q)
        // Track state is defined at z = 229 in the tilted sector frame
        float[] stateMean = new float[]{
                -31.87197f, -0.123296f, -0.13007024f, 0.001625887f, 0.348169f
        };
        float[] stateStd = new float[]{
                30.66571f, 18.169865f, 0.14433861f, 0.0830202f, 0.7989562f
        };

        // ------------------------------------------------------------
        // Translator: converts raw Java arrays to NDArrays and back
        Translator<float[][], float[]> translator = new Translator<>() {

            @Override
            public NDList processInput(TranslatorContext ctx, float[][] hits) {
                NDManager manager = ctx.getNDManager();
                int numHits = hits.length;

                // Normalize hits using training statistics
                float[][] normHits = new float[numHits][5];
                for (int i = 0; i < numHits; i++) {
                    for (int j = 0; j < 5; j++) {
                        normHits[i][j] = (hits[i][j] - hitMean[j]) / hitStd[j];
                    }
                }

                // Shape expected by the TorchScript model: [B, N, 5]
                // Here B = 1 (single track)
                NDArray x = manager.create(normHits)
                                   .reshape(1, numHits, 5);
                return new NDList(x);
            }

            @Override
            public float[] processOutput(TranslatorContext ctx, NDList list) {
                // Model output shape: [1, 5]
                NDArray out = list.get(0);
                float[] yNorm = out.toFloatArray(); // normalized output

                // De-normalize back to physical units
                float[] yDenorm = new float[yNorm.length];
                for (int i = 0; i < yNorm.length; i++) {
                    yDenorm[i] = yNorm[i] * stateStd[i] + stateMean[i];
                }
                return yDenorm;
            }

            @Override
            public Batchifier getBatchifier() {
                // No batching: one track per inference call
                return null;
            }
        };

        // ------------------------------------------------------------
        // Load TorchScript model using DJL Criteria
        Criteria<float[][], float[]> criteria = Criteria.builder()
                .setTypes(float[][].class, float[].class)
                .optModelPath(Paths.get("nets/transformer_default.pt"))
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<float[][], float[]> model = criteria.loadModel();
             Predictor<float[][], float[]> predictor = model.newPredictor()) {

            // ------------------------------------------------------------
            // Example input: variable-length list of hits
            // Each hit = (doca, xm, xr, yr, z)
            float[][] hitsExample = new float[][]{
                {0.1839f,-57.0643f,-54.8026f,21.8023f,229.2739f},
                {0.1472f,-57.7359f,-55.4817f,21.7297f,230.4328f},
                {0.4782f,-58.4075f,-56.1609f,21.6571f,231.5918f},
                {0.4782f,-57.7343f,-55.4194f,22.3153f,232.7498f},
                {0.1472f,-58.4059f,-56.0986f,22.2427f,233.9087f},
                // ... (remaining hits omitted for brevity)
            };

            // Run inference
            float[] predState = predictor.predict(hitsExample);

            // ------------------------------------------------------------
            // Print predicted track state
            System.out.printf("Predicted track state:%n");
            System.out.printf(
                    "x = %.4f, y = %.4f, tx = %.4f, ty = %.4f, Q = %.4f%n",
                    predState[0], predState[1], predState[2], predState[3], predState[4]
            );
        }
    }
}
