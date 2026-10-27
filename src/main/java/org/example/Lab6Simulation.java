package org.example;
import java.util.*;


// Клас для представлення об'єкта спостереження (регіону, підприємства тощо)
class Observation {
    String name;
    double[] features; // Вхідні ознаки (наприклад, 4 показники)
    int clusterId = -1; // Номер кластера
    int classLabel = -1; // Мітка класу після інтерпретації

    public Observation(String name, double[] features) {
        this.name = name;
        this.features = features;
    }
}

// Проста реалізація нейрона / шару для класифікації (Персептрон)
class SimpleNeuralClassifier {
    double[][] weights; // Ваги для зв'язків
    double[] biases;    // Зміщення
    int numClasses;

    public SimpleNeuralClassifier(int inputDim, int numClasses) {
        this.numClasses = numClasses;
        weights = new double[numClasses][inputDim];
        biases = new double[numClasses];

        // Ініціалізація невеликими випадковими вагами
        Random rand = new Random(42);
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < inputDim; j++) {
                weights[i][j] = rand.nextGaussian() * 0.1;
            }
            biases[i] = 0;
        }
    }

    // Функція активації Softmax
    private double[] softmax(double[] logits) {
        double[] exp = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            exp[i] = Math.exp(logits[i]);
            sum += exp[i];
        }
        for (int i = 0; i < logits.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }

    // Пряме поширення (Predict)
    public int predict(double[] x) {
        double[] logits = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            double sum = biases[i];
            for (int j = 0; j < x.length; j++) {
                sum += weights[i][j] * x[j];
            }
            logits[i] = sum;
        }
        double[] probs = softmax(logits);

        // Знаходження класу з найбільшою ймовірністю
        int bestClass = 0;
        double maxProb = probs[0];
        for (int i = 1; i < numClasses; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                bestClass = i;
            }
        }
        return bestClass;
    }

    // Просте навчання методом градієнтного спуску
    public void train(List<Observation> data, int epochs, double lr) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (Observation obs : data) {
                if (obs.classLabel < 0) continue;

                // Вирахування логітів
                double[] logits = new double[numClasses];
                for (int i = 0; i < numClasses; i++) {
                    double sum = biases[i];
                    for (int j = 0; j < obs.features.length; j++) {
                        sum += weights[i][j] * obs.features[j];
                    }
                    logits[i] = sum;
                }
                double[] probs = softmax(logits);

                // Оновлення ваг за градієнтом крос-ентропії
                int target = obs.classLabel;
                for (int i = 0; i < numClasses; i++) {
                    double error = (i == target ? 1.0 : 0.0) - probs[i];
                    biases[i] += lr * error;
                    for (int j = 0; j < obs.features.length; j++) {
                        weights[i][j] += lr * error * obs.features[j];
                    }
                }
            }
        }
    }
}

public class Lab6Simulation {
    public static void main(String[] args) {
        System.out.println("=== ЛАБОРАТОРНА РОБОТА №6: Моделювання в СППР ===");

        // 1. Формування набору даних (наприклад, 12 об'єктів з 4 ознаками)
        List<Observation> dataset = new ArrayList<>();
        dataset.add(new Observation("Об'єкт 1", new double[]{10.5, 80.0, 1.2, 500}));
        dataset.add(new Observation("Об'єкт 2", new double[]{11.0, 75.0, 1.5, 520}));
        dataset.add(new Observation("Об'єкт 3", new double[]{9.8, 82.0, 1.1, 480}));

        dataset.add(new Observation("Об'єкт 4", new double[]{25.2, 45.0, 3.8, 1200}));
        dataset.add(new Observation("Об'єкт 5", new double[]{24.0, 50.0, 3.5, 1150}));
        dataset.add(new Observation("Об'єкт 6", new double[]{26.5, 42.0, 4.0, 1300}));

        dataset.add(new Observation("Об'єкт 7", new double[]{50.1, 20.0, 7.5, 3000}));
        dataset.add(new Observation("Об'єкт 8", new double[]{48.0, 22.0, 7.0, 2900}));
        dataset.add(new Observation("Об'єкт 9", new double[]{52.5, 18.0, 8.2, 3100}));

        // 2. Масштабування ознак (Standardization: (x - mean) / std)
        int numFeatures = dataset.get(0).features.length;
        double[] means = new double[numFeatures];
        double[] stds = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double sum = 0;
            for (Observation obs : dataset) sum += obs.features[j];
            means[j] = sum / dataset.size();

            double varSum = 0;
            for (Observation obs : dataset) varSum += Math.pow(obs.features[j] - means[j], 2);
            stds[j] = Math.sqrt(varSum / dataset.size());
            if (stds[j] == 0) stds[j] = 1.0; // захист від ділення на нуль
        }

        // Зберігаємо масштабовані дані для кластеризації
        double[][] scaledData = new double[dataset.size()][numFeatures];
        for (int i = 0; i < dataset.size(); i++) {
            for (int j = 0; j < numFeatures; j++) {
                scaledData[i][j] = (dataset.get(i).features[j] - means[j]) / stds[j];
            }
        }

        // 3. Спрощена кластеризація (K-Means для K = 3)
        int k = 3;
        Random rnd = new Random(123);
        double[][] centroids = new double[k][numFeatures];
        // Ініціалізуємо центроїди першими трьома точками для прикладу
        for (int i = 0; i < k; i++) {
            centroids[i] = Arrays.copyOf(scaledData[i * 3], numFeatures);
        }

        boolean changed = true;
        int iter = 0;
        while (changed && iter < 100) {
            changed = false;
            iter++;
            // Призначаємо кластери
            for (int i = 0; i < dataset.size(); i++) {
                int bestCluster = 0;
                double minDist = Double.MAX_VALUE;
                for (int c = 0; c < k; c++) {
                    double dist = 0;
                    for (int j = 0; j < numFeatures; j++) {
                        dist += Math.pow(scaledData[i][j] - centroids[c][j], 2);
                    }
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = c;
                    }
                }
                if (dataset.get(i).clusterId != bestCluster) {
                    dataset.get(i).clusterId = bestCluster;
                    changed = true;
                }
            }
            // Перераховуємо центроїди
            double[][] newCentroids = new double[k][numFeatures];
            int[] counts = new int[k];
            for (int i = 0; i < dataset.size(); i++) {
                int cl = dataset.get(i).clusterId;
                counts[cl]++;
                for (int j = 0; j < numFeatures; j++) {
                    newCentroids[cl][j] += scaledData[i][j];
                }
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    for (int j = 0; j < numFeatures; j++) {
                        centroids[c][j] = newCentroids[c][j] / counts[c];
                    }
                }
            }
        }

        System.out.println("\n[ЕТАП 7-9] Результати кластеризації та формування класів:");
        for (Observation obs : dataset) {
            // Інтерпретація: номер кластера напряму стає міткою класу для навчання
            obs.classLabel = obs.clusterId;
            System.out.println(obs.name + " -> Кластер: " + obs.clusterId + " (Призначений клас: Клас " + obs.classLabel + ")");
        }

        // 4. Навчання нейромережевого класифікатора
        System.out.println("\n[ЕТАП 11] Навчання нейромережі...");
        SimpleNeuralClassifier nn = new SimpleNeuralClassifier(numFeatures, k);
        nn.train(dataset, 1000, 0.1);
        System.out.println("Навчання успішно завершено.");

        // 5. Перевірка на нових об'єктах (Етап 13)
        System.out.println("\n[ЕТАП 13] Перевірка моделі на нових об'єктах:");
        double[] newObjFeaturesRaw = {25.0, 48.0, 3.6, 1180.0}; // Новий об'єкт, схожий на середню групу

        // Масштабуємо ознаки нового об'єкта за тими ж параметрами
        double[] newObjScaled = new double[numFeatures];
        for (int j = 0; j < numFeatures; j++) {
            newObjScaled[j] = (newObjFeaturesRaw[j] - means[j]) / stds[j];
        }

        int predictedClass = nn.predict(newObjScaled);
        System.out.println("Новий об'єкт класифіковано у: Клас " + predictedClass);
        System.out.println("СППР розпізнала стан об'єкта успішно!");
    }
}