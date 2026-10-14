
public class SampleData {
    private double[] features;   // Вхідні ознаки (параметри для аналізу)
    private String targetClass;    // Цільовий клас (результат класифікації / діагноз)
    private double targetValue;    // Чіслове значення для прогнозу

    public SampleData(double[] features, String targetClass, double targetValue) {
        this.features = features;
        this.targetClass = targetClass;
        this.targetValue = targetValue;
    }

    public double[] getFeatures() {return features;}
    public String getTargetClass() {return targetClass;}
    public double getTargetValue() {return targetValue;}
}