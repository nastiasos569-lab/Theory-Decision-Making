package org.example;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.io.File;
import java.util.Random;

public class Lab5NeuralNetwork {
    public static void main(String[] args) {
        try {
            // Крок 1: Завантаження даних із CSV-файлу
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File("dataset.csv"));
            Instances data = loader.getDataSet();

            // Встановлюємо останній стовпець як цільову змінну (Y)
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            System.out.println("=== ПАСПОРТ ТА СТАТИСТИКА ДАНИХ ===");
            System.out.println("Загальна кількість спостережень (рядків): " + data.numInstances());
            System.out.println("Кількість ознак (факторів + ціль): " + data.numAttributes());

            // Крок 2: Попереднє опрацювання та нормалізація даних
            Normalize normalizer = new Normalize();
            normalizer.setInputFormat(data);
            Instances normalizedData = Filter.useFilter(data, normalizer);
            System.out.println("\n[OK] Дані успішно нормалізовано.");

            // Крок 3: Перемішування та поділ на навчальну (70%) та тестову (30%) множини (пункт 9, 14)
            normalizedData.randomize(new Random(42)); // 42 — зерно для генератора випадкових чисел (для відтворюваності)

            int trainSize = (int) Math.round(normalizedData.numInstances() * 0.7);
            int testSize = normalizedData.numInstances() - trainSize;

            Instances trainData = new Instances(normalizedData, 0, trainSize);
            Instances testData = new Instances(normalizedData, trainSize, testSize);

            System.out.println("Розмір навчальної вибірки: " + trainData.numInstances());
            System.out.println("Розмір тестової вибірки: " + testData.numInstances());

            // Крок 4: Створення та налаштування нейромережі MLP (Багатошаровий перцептрон, пункт 10)
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setLearningRate(0.3);      // Швидкість навчання
            mlp.setMomentum(0.2);          // Моментум
            mlp.setTrainingTime(500);      // Кількість епох навчання
            mlp.setHiddenLayers("a");      // "a" — автоматичний вибір кількості нейронів у прихованому шарі

            // Крок 5: Навчання моделі (пункт 10, 14)
            System.out.println("\nБудується та навчається нейромережева модель (MLP)...");
            mlp.buildClassifier(trainData);

            // Крок 6: Оцінювання якості моделі (пункт 11)
            Evaluation evalTrain = new Evaluation(trainData);
            evalTrain.evaluateModel(mlp, trainData);

            Evaluation evalTest = new Evaluation(trainData);
            evalTest.evaluateModel(mlp, testData);

            // Крок 7: Виведення результатів для таблиці у звіті (пункт 10, 15)
            System.out.println("\n=== РЕЗУЛЬТАТИ МОДЕЛЮВАННЯ (ТАБЛИЦЯ ЯКОСТІ) ===");
            System.out.println("Показник\t\tНавчальна вибірка\tТестова вибірка");
            System.out.println("RMSE (Помилка):\t\t" + String.format("%.4f", evalTrain.rootMeanSquaredError()) + "\t\t" + String.format("%.4f", evalTest.rootMeanSquaredError()));
            System.out.println("MAE (Сер. абс. похибка): " + String.format("%.4f", evalTrain.meanAbsoluteError()) + "\t\t" + String.format("%.4f", evalTest.meanAbsoluteError()));

        } catch (Exception e) {
            System.err.println("Помилка під час виконання програми: " + e.getMessage());
            e.printStackTrace();
        }
    }
}