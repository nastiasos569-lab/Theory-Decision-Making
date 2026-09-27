import java.util.*;

public class LabEnsembleApp {

    // Клас для об'єкта класифікації
    static class ClassInstance {
        double[] features;
        int label;
        public ClassInstance(double[] features, int label) {
            this.features = features;
            this.label = label;
        }
    }

    // Клас для об'єкта регресії
    static class RegInstance {
        double[] features;
        double target;
        public RegInstance(double[] features, double target) {
            this.features = features;
            this.target = target;
        }
    }

    public static void main(String[] args) {
        System.out.println("ВИКОНАННЯ ЛАБОРАТОРНОЇ РОБОТИ №1 В JAVA");
        runClassificationTask();
        runRegressionTask();
    }

    // --- ЧАСТИНА 1: КЛАСИФІКАЦІЯ ---
    public static void runClassificationTask() {
        System.out.println("\n[ПРИКЛАД 1] Завдання класифікації");

        // Вхідні дані з лабораторної (30 об'єктів: 10 класу 1, 10 класу 2, 10 класу 3)
        double[][] classData = {
                {1.7, 0.84, 1.69, 0.61, 1}, {1.83, 1.75, 2.55, 0.42, 1}, {1.39, 0.84, 1.61, 1.01, 1}, {0.84, 1.08, 2.21, 1.06, 1}, {1.11, 1.62, 2.43, 0.31, 1},
                {1.93, 1.78, 1.78, 0.52, 1}, {2.14, 1.27, 1.43, 0.44, 1}, {1.99, 1.65, 2.43, 1.32, 1}, {1.55, 2.16, 1.83, 1.07, 1}, {1.96, 1.67, 2.51, 1.11, 1},
                {4.79, 3.36, 3.12, 4.21, 2}, {3.91, 3.63, 2.94, 4.19, 2}, {4.69, 3.81, 3.32, 4.09, 2}, {4.17, 4.61, 3.71, 4.65, 2}, {4.04, 4.32, 3.03, 4.33, 2},
                {5.19, 4.2, 3.58, 4.76, 2}, {4.98, 4.39, 3.12, 3.84, 2}, {4.24, 3.67, 3.1, 5.12, 2}, {5.03, 3.74, 3.72, 4.35, 2}, {5.08, 3.94, 3.17, 4.15, 2},
                {7.09, 6.17, 7.12, 6.56, 3}, {6.86, 6.11, 7.7, 6.01, 3}, {6.43, 5.87, 6.45, 6.18, 3}, {7.41, 6.39, 6.39, 5.83, 3}, {7.69, 6.54, 7.66, 6.51, 3},
                {6.32, 6.81, 7.25, 6.05, 3}, {6.67, 6.7, 6.46, 5.91, 3}, {6.94, 7.14, 7.53, 5.67, 3}, {7.0, 6.05, 7.58, 6.52, 3}, {6.72, 6.69, 7.15, 5.51, 3}
        };

        List<ClassInstance> dataset = new ArrayList<>();
        for (double[] row : classData) {
            double[] features = Arrays.copyOfRange(row, 0, 4);
            int label = (int) row[4];
            dataset.add(new ClassInstance(features, label));
        }

        // Розподіл на навчальну (70%) та тестову (30%) вибірки
        int splitIdx = (int) (dataset.size() * 0.7);
        List<ClassInstance> trainSet = dataset.subList(0, splitIdx);
        List<ClassInstance> testSet = dataset.subList(splitIdx, dataset.size());

        // Експерти з різною складністю (параметр k):
        // E1: k=1 (мала складність / перенавчання)
        // E2: k=3 (оптимальна складність)
        // E3: k=5 (велика складність / згладжування)

        double correctE1 = 0, correctE2 = 0, correctE3 = 0, correctEns = 0;

        System.out.println("Об'єкт | E1 (k=1) | E2 (k=3) | E3 (k=5) | Колектив | Реал.");
        for (int i = 0; i < testSet.size(); i++) {
            ClassInstance testObj = testSet.get(i);

            int e1Pred = classifyKNN(trainSet, testObj.features, 1);
            int e2Pred = classifyKNN(trainSet, testObj.features, 3);
            int e3Pred = classifyKNN(trainSet, testObj.features, 5);

            // Агрегування: мажоритарне голосування (Majority Voting)
            int ensemblePred = majorityVote(e1Pred, e2Pred, e3Pred);

            if (e1Pred == testObj.label) correctE1++;
            if (e2Pred == testObj.label) correctE2++;
            if (e3Pred == testObj.label) correctE3++;
            if (ensemblePred == testObj.label) correctEns++;

            System.out.printf("  %2d   |    %d     |    %d     |    %d     |    %d     |   %d   \n",
                    (i+1), e1Pred, e2Pred, e3Pred, ensemblePred, testObj.label);
        }

        System.out.printf("\nТочність (Accuracy) моделей на тестовій вибірці:\n");
        System.out.printf("E1 (k=1): %.2f%%\n", (correctE1 / testSet.size()) * 100);
        System.out.printf("E2 (k=3): %.2f%%\n", (correctE2 / testSet.size()) * 100);
        System.out.printf("E3 (k=5): %.2f%%\n", (correctE3 / testSet.size()) * 100);
        System.out.printf("Колектив:  %.2f%%\n", (correctEns / testSet.size()) * 100);
    }

    private static int classifyKNN(List<ClassInstance> trainSet, double[] query, int k) {
        class Neighbor {
            double dist;
            int label;
            Neighbor(double d, int l) { dist = d; label = l; }
        }
        List<Neighbor> neighbors = new ArrayList<>();
        for (ClassInstance trainObj : trainSet) {
            double dist = 0;
            for (int i = 0; i < query.length; i++) {
                dist += Math.pow(query[i] - trainObj.features[i], 2);
            }
            neighbors.add(new Neighbor(Math.sqrt(dist), trainObj.label));
        }
        neighbors.sort(Comparator.comparingDouble(n -> n.dist));

        Map<Integer, Integer> voteCount = new HashMap<>();
        for (int i = 0; i < Math.min(k, neighbors.size()); i++) {
            int lbl = neighbors.get(i).label;
            voteCount.put(lbl, voteCount.getOrDefault(lbl, 0) + 1);
        }
        return Collections.max(voteCount.entrySet(), Map.Entry.comparingByValue()).getKey();
    }

    private static int majorityVote(int e1, int e2, int e3) {
        if (e1 == e2 || e1 == e3) return e1;
        if (e2 == e3) return e2;
        return e1;
    }

    // --- ЧАСТИНА 2: РЕГРЕСІЯ ---
    public static void runRegressionTask() {
        System.out.println("\n[ПРИКЛАД 2] Багатофакторна регресія");

        double[][] regData = {
                {0.39, 6.96, 1.44, 4.63, 9.98}, {4.53, 4.98, 0.19, 4.32, 15.67}, {3.72, 8.52, 5.47, 7.55, 17.24}, {8.51, 8.2, 3.76, 0.96, 25.45}, {5.12, 4.96, 7.72, 5.97, 11.94},
                {3.77, 0.49, 0.61, 9.42, 12.33}, {8.39, 2.96, 8.89, 5.22, 18.02}, {5.84, 9.33, 6.79, 0.94, 20.27}, {6.23, 7.87, 8.99, 3.31, 15.32}, {1.67, 8.48, 7.81, 0.5, 8.03},
                {5.91, 3.68, 1.86, 6.04, 19.64}, {1.13, 4.95, 7.79, 9.55, 6.95}, {7.15, 0.41, 9.67, 6.02, 10.66}, {0.86, 6.65, 1.54, 8.07, 13.49}, {1.09, 5.35, 4.43, 0.38, 8.38},
                {9.77, 1.24, 7.35, 9.89, 20.26}, {9.04, 8.85, 3.38, 7.17, 27.64}, {7.03, 8.44, 8.57, 3.86, 20.13}, {7.12, 0.93, 2.06, 9.37, 16.98}, {5.76, 2.6, 5.86, 9.12, 15.8},
                {3.27, 3.47, 1.86, 1.78, 11.34}, {9.65, 6.7, 9.59, 2.22, 19.78}, {1.88, 2.17, 3.84, 9.9, 8.66}, {0.61, 0.27, 5.46, 6.13, 1.59}, {5.11, 9.15, 8.61, 9.51, 19.15},
                {1.3, 7.64, 1.55, 0.06, 10.88}, {6.99, 1.25, 6.22, 2.01, 11.44}, {3.76, 3.28, 2.61, 1.34, 10.47}, {5.76, 6.93, 2.0, 0.44, 18.93}, {2.73, 6.48, 0.74, 7.03, 18.66}
        };

        List<RegInstance> dataset = new ArrayList<>();
        for (double[] row : regData) {
            double[] features = Arrays.copyOfRange(row, 0, 4);
            double target = row[4];
            dataset.add(new RegInstance(features, target));
        }

        int splitIdx = (int) (dataset.size() * 0.7);
        List<RegInstance> trainSet = dataset.subList(0, splitIdx);
        List<RegInstance> testSet = dataset.subList(splitIdx, dataset.size());

        double maeE1 = 0, maeE2 = 0, maeE3 = 0, maeEns = 0;
        double rmseE1 = 0, rmseE2 = 0, rmseE3 = 0, rmseEns = 0;

        System.out.println("Об'єкт | E1 (k=1) | E2 (k=3) | E3 (k=5) | Колектив | Реал (Y)");
        for (int i = 0; i < testSet.size(); i++) {
            RegInstance testObj = testSet.get(i);

            double e1Pred = predictKNN(trainSet, testObj.features, 1);
            double e2Pred = predictKNN(trainSet, testObj.features, 3);
            double e3Pred = predictKNN(trainSet, testObj.features, 5);

            // Агрегування: усереднення прогнозів
            double ensemblePred = (e1Pred + e2Pred + e3Pred) / 3.0;

            maeE1 += Math.abs(e1Pred - testObj.target);
            maeE2 += Math.abs(e2Pred - testObj.target);
            maeE3 += Math.abs(e3Pred - testObj.target);
            maeEns += Math.abs(ensemblePred - testObj.target);

            rmseE1 += Math.pow(e1Pred - testObj.target, 2);
            rmseE2 += Math.pow(e2Pred - testObj.target, 2);
            rmseE3 += Math.pow(e3Pred - testObj.target, 2);
            rmseEns += Math.pow(ensemblePred - testObj.target, 2);

            System.out.printf("  %2d   |  %6.2f  |  %6.2f  |  %6.2f  |  %6.2f  |  %6.2f \n",
                    (i+1), e1Pred, e2Pred, e3Pred, ensemblePred, testObj.target);
        }

        int n = testSet.size();
        System.out.printf("\nПоказники якості (MAE / RMSE) на тестовій вибірці:\n");
        System.out.printf("Експерт  | MAE    | RMSE\n");
        System.out.printf("E1 (k=1) | %.2f   | %.2f\n", maeE1/n, Math.sqrt(rmseE1/n));
        System.out.printf("E2 (k=3) | %.2f   | %.2f\n", maeE2/n, Math.sqrt(rmseE2/n));
        System.out.printf("E3 (k=5) | %.2f   | %.2f\n", maeE3/n, Math.sqrt(rmseE3/n));
        System.out.printf("Колектив | %.2f   | %.2f\n", maeEns/n, Math.sqrt(rmseEns/n));
    }

    private static double predictKNN(List<RegInstance> trainSet, double[] query, int k) {
        class Neighbor {
            double dist;
            double target;
            Neighbor(double d, double t) { dist = d; target = t; }
        }
        List<Neighbor> neighbors = new ArrayList<>();
        for (RegInstance trainObj : trainSet) {
            double dist = 0;
            for (int i = 0; i < query.length; i++) {
                dist += Math.pow(query[i] - trainObj.features[i], 2);
            }
            neighbors.add(new Neighbor(Math.sqrt(dist), trainObj.target));
        }
        neighbors.sort(Comparator.comparingDouble(n -> n.dist));

        double sum = 0;
        int count = Math.min(k, neighbors.size());
        for (int i = 0; i < count; i++) {
            sum += neighbors.get(i).target;
        }
        return sum / count;
    }
}