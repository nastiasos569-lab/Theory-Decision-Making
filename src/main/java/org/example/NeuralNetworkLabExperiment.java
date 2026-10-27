package org.example;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;

public class NeuralNetworkLabExperiment {

    public static void main(String[] args) {
        System.out.println("=== Дослідження впливу алгоритму навчання нейромережі на продуктивність моделі ===");

        // 1. Підготовка даних (моделюємо задачу класифікації)
        int numRows = 1000;
        int numInputs = 4;
        int numOutputs = 3;

        INDArray features = Nd4j.rand(numRows, numInputs);
        INDArray labels = Nd4j.zeros(numRows, numOutputs);
        for (int i = 0; i < numRows; i++) {
            int classIdx = (int) (Math.random() * numOutputs);
            labels.putScalar(i, classIdx, 1.0);
        }

        DataSet dataSet = new DataSet(features, labels);
        int epochs = 50; // Фіксуємо кількість епох[cite: 1]

        // 2. Проведення експериментів з різними алгоритмами та гіперпараметрами[cite: 1]

        System.out.println("\n--- Експеримент 1: Зворотне поширення (Базовий режим SGD) ---[cite: 1]");
        runAndEvaluate("Базовий SGD (lr = 0.01)", new Sgd(0.01), dataSet, epochs);

        System.out.println("\n--- Експеримент 2: Вплив швидкості навчання (Learning Rate) ---[cite: 1]");
        runAndEvaluate("Занадто мала швидкість (lr = 0.0001)", new Sgd(0.0001), dataSet, epochs);
        runAndEvaluate("Занадто велика швидкість (lr = 0.5)", new Sgd(0.5), dataSet, epochs);

        System.out.println("\n--- Експеримент 3: Вплив моменту (Momentum / Nesterovs) ---[cite: 1]");
        runAndEvaluate("SGD + Момент (lr = 0.01, momentum = 0.9)", new Nesterovs(0.01, 0.9), dataSet, epochs);

        System.out.println("\n--- Експерименти 4-6: Альтернативні методи оптимізації ---[cite: 1]");
        runAndEvaluate("Adam Optimizer (адаптивний метод)", new Adam(0.01), dataSet, epochs);
    }

    /**
     * Метод для створення, навчання моделі та вимірювання її продуктивності
     */
    public static void runAndEvaluate(String modelName, org.nd4j.linalg.learning.config.IUpdater updater, DataSet ds, int epochs) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(updater)
                .list()
                .layer(new DenseLayer.Builder().nIn(ds.numInputs()).nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(10).nOut(ds.numOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        long startTime = System.currentTimeMillis();
        double finalScore = 0.0;

        for (int e = 0; e < epochs; e++) {
            model.fit(ds);
            finalScore = model.score();
        }

        long endTime = System.currentTimeMillis();
        long trainingTime = endTime - startTime;

        // Оцінка якості через прогнозування та клас Evaluation
        INDArray predicted = model.output(ds.getFeatures());
        Evaluation eval = new Evaluation(ds.numOutcomes());
        eval.eval(ds.getLabels(), predicted);
        double accuracy = eval.accuracy();

        // Виведення результатів
        System.out.println(" Конфігурація: " + modelName);
        System.out.println("   -> Час навчання: " + trainingTime + " мс");
        System.out.println("   -> Епохи: " + epochs);
        System.out.printf("   -> Помилка (Score): %.4f\n", finalScore);
        System.out.printf("   -> Точність (Accuracy): %.2f%%\n\n", accuracy * 100);
    }
}