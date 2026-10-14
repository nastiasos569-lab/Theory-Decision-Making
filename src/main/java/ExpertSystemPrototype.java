import java.util.ArrayList;
import java.util.List;

public class ExpertSystemPrototype {
    private String version;                     // Назва версії прототипу (напр., ЕС-1, ЕС-2)
    private List<SampleData> knowledgeBase;     // База знань (навчальна вибірка)

    public ExpertSystemPrototype(String version) {
        this.version = version;
        this.knowledgeBase = new ArrayList<>();
    }


    public void addData(double[] features, String targetClass, double targetValue) {
        knowledgeBase.add(new SampleData(features, targetClass, targetValue));
    }

    /**
     * Метод класифікації вхідних даних (імітація нейромережевого висновку).
     * Використовує алгоритм пошуку найближчого сусіда (KNN).
     * @param inputFeatures вхідні параметри від користувача
     * @return результат класифікації (клас)
     */
    public String classify(double[] inputFeatures) {
        if (knowledgeBase.isEmpty()) {
            return "База знань порожня";
        }

        // Знаходимо найближчий елемент у базі знань
        SampleData bestMatch = knowledgeBase.get(0);
        double minDistance = calculateDistance(inputFeatures, bestMatch.getFeatures());

        for (int i = 1; i < knowledgeBase.size(); i++) {
            SampleData current = knowledgeBase.get(i);
            double dist = calculateDistance(inputFeatures, current.getFeatures());
            if (dist < minDistance) {
                minDistance = dist;
                bestMatch = current;
            }
        }
        return bestMatch.getTargetClass();
    }

    /**
     * Допоміжний метод для розрахунку евклидової відстані між векторами ознак.
     */
    private double calculateDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    /**
     * Верифікація та оцінка точності моделі (перевірка адекватності і надійності).
     * @param testSet тестова вибірка даних
     * @return відсоток правильно розпізнаних випадків (точність)
     */
    public double evaluateAccuracy(List<SampleData> testSet) {
        int correct = 0;
        for (SampleData test : testSet) {
            String predicted = classify(test.getFeatures());
            if (predicted.equals(test.getTargetClass())) {
                correct++;
            }
        }
        return (double) correct / testSet.size() * 100.0;
    }

    /**
     * Повертає назву версії прототипу.
     */
    public String getVersion() {
        return version;
    }
}