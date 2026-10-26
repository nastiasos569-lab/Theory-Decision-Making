package org.example;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.util.Random;

public class DecisionSystemLab4 {

    public static void main(String[] args) {
        try {
            // 1. Завантаження даних (формат ARFF)
            File file = new File("dataset.arff");
            ArffLoader loader = new ArffLoader();
            loader.setFile(file);
            Instances data = loader.getDataSet();

            // Встановлюємо цільову змінну
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            System.out.println("Загальна кількість об'єктів: " + data.numInstances());

            // 2. Поділ на робочу (навчальну) та тестову вибірки (наприклад, 70% / 30%)
            int trainSize = (int) Math.round(data.numInstances() * 0.7);
            int testSize = data.numInstances() - trainSize;

            data.randomize(new Random(1)); // Перемішування даних
            Instances trainData = new Instances(data, 0, trainSize);
            Instances testData = new Instances(data, trainSize, testSize);

            // 3. Побудова альтернативних моделей
            // Модель 1: Дерево рішень (J48)
            J48 treeModel = new J48();
            treeModel.buildClassifier(trainData);

            // Модель 2: Штучна нейронна мережа (MultilayerPerceptron)
            MultilayerPerceptron nnModel = new MultilayerPerceptron();
            nnModel.setLearningRate(0.3);
            nnModel.setTrainingTime(200);
            nnModel.buildClassifier(trainData);

            // 4. Оцінка якості моделей на тестовій вибірці
            Evaluation evalTree = new Evaluation(trainData);
            evalTree.evaluateModel(treeModel, testData);

            System.out.println("\n--- Результати моделі 1 (Дерево рішень) ---");
            System.out.println("Точність (Accuracy): " + (100 - evalTree.pctIncorrect()) + "%");
            System.out.println("Кількість помилок: " + evalTree.incorrect());

            Evaluation evalNN = new Evaluation(trainData);
            evalNN.evaluateModel(nnModel, testData);

            System.out.println("\n--- Результати моделі 2 (Нейронна мережа) ---");
            // Виправлено: додано знак мінус (-) для обчислення точності
            System.out.println("Точність (Accuracy): " + (100 - evalNN.pctIncorrect()) + "%");
            System.out.println("Кількість помилок: " + evalNN.incorrect());

            // 5. Аналіз помилок I та II роду (Пункт 6.4)
            // Матриця плутанини (Confusion Matrix):
            // рядки - фактичні класи, стовпчики - передбачені моделлю
            double[][] confusionMatrix = evalNN.confusionMatrix();
            System.out.println("\n--- Матриця помилок (Confusion Matrix) для НН ---");
            for (int i = 0; i < confusionMatrix.length; i++) {
                for (int j = 0; j < confusionMatrix[i].length; j++) {
                    System.out.print((int) confusionMatrix[i][j] + "\t");
                }
                System.out.println();
            }

            defineAndApplyLossMatrix(nnModel, testData);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Метод для демонстрації ризик-орієнтованого правила та матриці втрат
     */
    public static void defineAndApplyLossMatrix(MultilayerPerceptron model, Instances testData) throws Exception {
        System.out.println("\n--- Аналіз з урахуванням матриці втрат ---");

        double[][] lossMatrix = {
                { 0,  2,  10 }, // Фактичний клас 1
                { 5,  0,   4 }, // Фактичний клас 2
                { 20, 8,   0 }  // Фактичний клас 3
        };

        double totalStandardLoss = 0;
        double totalRiskOptimizedLoss = 0;

        for (int i = 0; i < testData.numInstances(); i++) {
            Instance inst = testData.instance(i);
            double actualClass = inst.classValue();

            double[] probabilities = model.distributionForInstance(inst);

            int bestStandardClass = 0;
            double maxProb = probabilities[0];
            for (int c = 1; c < probabilities.length; c++) {
                if (probabilities[c] > maxProb) {
                    maxProb = probabilities[c];
                    bestStandardClass = c;
                }
            }

            int bestRiskClass = 0;
            double minExpectedLoss = Double.MAX_VALUE;

            for (int m = 0; m < probabilities.length; m++) {
                double expectedLoss = 0;
                for (int n = 0; n < probabilities.length; n++) {
                    expectedLoss += lossMatrix[m][n] * probabilities[n];
                }
                if (expectedLoss < minExpectedLoss) {
                    minExpectedLoss = expectedLoss;
                    bestRiskClass = m;
                }
            }

            totalStandardLoss += lossMatrix[bestStandardClass][(int) actualClass];
            totalRiskOptimizedLoss += lossMatrix[bestRiskClass][(int) actualClass];
        }

        System.out.println("Сумарні втрати за стандартним правилом (max ймовірність): " + totalStandardLoss);
        System.out.println("Сумарні втрати за ризик-орієнтованим правилом (min збиток): " + totalRiskOptimizedLoss);
    }
}