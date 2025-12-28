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

public class Main {

    public static void main(String[] args) throws IOException, ModelException, TranslateException {

        // ------------------------------
        // Hits normalization参数 (mean / std)
        float[] hitMean = new float[]{0.54767877f, -39.37779f, -39.37546f, 60.905655f, 374.25867f};
        float[] hitStd  = new float[]{0.4348458f, 51.83236f, 52.26446f, 35.65593f, 111.37744f};

        // ------------------------------
        // State 反归一化参数 (mean / std)
        float[] stateMean = new float[]{-31.87197f, -0.123296f, -0.13007024f, 0.001625887f, 0.348169f};
        float[] stateStd  = new float[]{30.66571f, 18.169865f, 0.14433861f, 0.0830202f, 0.7989562f};

        // ------------------------------
        // Translator
        Translator<float[][], float[]> translator = new Translator<float[][], float[]>() {

            @Override
            public NDList processInput(TranslatorContext ctx, float[][] hits) {
                NDManager manager = ctx.getNDManager();
                int numHits = hits.length;

                // 归一化 hits
                float[][] normHits = new float[numHits][5];
                for (int i = 0; i < numHits; i++) {
                    for (int j = 0; j < 5; j++) {
                        normHits[i][j] = (hits[i][j] - hitMean[j]) / hitStd[j];
                    }
                }

                NDArray x = manager.create(normHits).reshape(1, numHits, 5); // [1, num_hits, 5]
                return new NDList(x);
            }

            @Override
            public float[] processOutput(TranslatorContext ctx, NDList list) {
                NDArray out = list.get(0); // shape [1, 5]
                float[] yNorm = out.toFloatArray(); // 归一化输出

                // 反归一化
                float[] yDenorm = new float[yNorm.length];
                for (int i = 0; i < yNorm.length; i++) {
                    yDenorm[i] = yNorm[i] * stateStd[i] + stateMean[i];
                }
                return yDenorm;
            }

            @Override
            public Batchifier getBatchifier() {
                return null; // 不使用批处理
            }
        };

        // ------------------------------
        // Load model
        Criteria<float[][], float[]> criteria = Criteria.builder()
                .setTypes(float[][].class, float[].class)
                .optModelPath(Paths.get("nets/tae_default.pt"))
                .optEngine("PyTorch")
                .optTranslator(translator)
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<float[][], float[]> model = criteria.loadModel();
             Predictor<float[][], float[]> predictor = model.newPredictor()) {

            // ------------------------------
            // 示例输入
            float[][] hitsExample = new float[][]{
                {0.1839f,-57.0643f,-54.8026f,21.8023f,229.2739f},
                {0.1472f,-57.7359f,-55.4817f,21.7297f,230.4328f},
                {0.4782f,-58.4075f,-56.1609f,21.6571f,231.5918f},
                {0.4782f,-57.7343f,-55.4194f,22.3153f,232.7498f},
                {0.1472f,-58.4059f,-56.0986f,22.2427f,233.9087f},
                {0.1839f,-59.0776f,-56.7777f,22.1701f,235.0677f},
                {0.1318f,-56.3986f,-58.7737f,22.3070f,239.9370f},
                {0.3826f,-57.1019f,-59.4697f,22.2387f,241.1501f},
                {0.4881f,-56.3969f,-58.8306f,22.8572f,242.3623f},
                {0.0264f,-57.1002f,-59.5266f,22.7889f,243.5754f},
                {0.5408f,-57.8034f,-60.2225f,22.7205f,244.7886f},
                {0.3299f,-57.0985f,-59.5835f,23.3390f,246.0007f},
                {0.5956f,-76.9856f,-73.2139f,36.7100f,353.7574f},
                {0.7198f,-78.0666f,-74.3073f,36.5894f,355.6241f},
                {0.0993f,-76.9820f,-73.1113f,37.6737f,357.4888f},
                {0.3971f,-76.9783f,-73.0087f,38.6373f,361.2202f},
                {0.4220f,-75.8937f,-71.8126f,39.7216f,363.0849f},
                {0.2151f,-70.0672f,-74.4737f,41.0035f,374.0818f},
                {0.6233f,-68.9180f,-73.4350f,42.0318f,376.0565f},
                {0.7954f,-70.0633f,-74.5681f,41.9172f,378.0334f},
                {0.0430f,-68.9141f,-73.5294f,42.9455f,380.0081f},
                {0.8814f,-67.7649f,-72.4907f,43.9738f,381.9827f},
                {0.5372f,-68.9103f,-73.6238f,43.8592f,383.9597f},
                {1.0528f,-42.8870f,-33.6681f,90.4026f,492.1759f},
                {2.1281f,-39.6308f,-30.2308f,92.1786f,492.1743f},
                {0.0443f,-41.3176f,-31.9391f,91.9671f,494.9806f},
                {2.0995f,-42.8844f,-33.5208f,91.8211f,497.7867f},
                {1.0814f,-39.6281f,-30.0834f,93.5971f,497.7852f},
                {1.0909f,-41.3149f,-31.7918f,93.3856f,500.5914f},
                {0.0348f,-39.6255f,-29.9361f,95.0155f,503.3960f},
                {1.0433f,-38.0560f,-28.2071f,96.5801f,506.2007f},
                {0.7538f,-24.1916f,-34.5060f,95.2968f,514.4334f},
                {0.0667f,-22.5464f,-33.0199f,96.7665f,517.3666f},
                {0.7344f,-20.7812f,-31.4201f,98.2950f,520.2997f},
                {1.4215f,-19.1360f,-29.9340f,99.7648f,523.2328f},
                {1.0113f,-20.7784f,-31.5613f,99.6259f,526.1676f},
                {0.3241f,-19.1332f,-30.0752f,101.0956f,529.1008f}
            };

            float[] predState = predictor.predict(hitsExample);

            System.out.printf("Predicted track state:%n");
            System.out.printf("x = %.4f, y = %.4f, tx = %.4f, ty = %.4f, Q = %.4f%n",
                    predState[0], predState[1], predState[2], predState[3], predState[4]);
        }
    }
}
