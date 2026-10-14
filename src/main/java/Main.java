import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        System.out.println("=== Лабораторна робота: Моделювання ЕС у базисі NN ===");

        ExpertSystemPrototype es1 = new ExpertSystemPrototype("ЕС-1 (Первинний прототип)");
        es1.addData(new double[]{0.2, 0.8}, "Норма", 10.0);
        es1.addData(new double[]{0.9, 0.1}, "Ризик", 85.5);
        es1.addData(new double[]{0.1, 0.9}, "Норма", 12.0);
        es1.addData(new double[]{0.85, 0.2}, "Ризик", 80.0);

        List<SampleData> testSet = Arrays.asList(
                new SampleData(new double[]{0.25, 0.75}, "Норма", 11.0),
                new SampleData(new double[]{0.8, 0.15}, "Ризик", 82.0)
        );

        double accuracy1 = es1.evaluateAccuracy(testSet);
        System.out.println("\n[Тестування] " + es1.getVersion());
        System.out.println("Точність (адекватність) моделі: " + accuracy1 + "%");

        ExpertSystemPrototype es2 = new ExpertSystemPrototype("ЕС-2 (Розширена промислова версія)");
        es2.addData(new double[]{0.2, 0.8}, "Норма", 10.0);
        es2.addData(new double[]{0.9, 0.1}, "Ризик", 85.5);
        es2.addData(new double[]{0.1, 0.9}, "Норма", 12.0);
        es2.addData(new double[]{0.85, 0.2}, "Ризик", 80.0);
        es2.addData(new double[]{0.5, 0.5}, "Помірний ризик", 50.0);

        double accuracy2 = es2.evaluateAccuracy(testSet);
        System.out.println("\n[Тестування] " + es2.getVersion());
        System.out.println("Точність розширеної моделі: " + accuracy2 + "%");

        // Етап експлуатації (Інтерактивний діалог)
        Scanner scanner = new Scanner(System.in);
        scanner.useLocale(Locale.US);

        System.out.println("\n--- Етап експлуатації (Інтерактивний діалог з користувачем) ---");
        System.out.print("Введіть значення першого параметра (від 0.0 до 1.0): ");
        double p1 = scanner.nextDouble();

        System.out.print("Введіть значення другого параметра (від 0.0 до 1.0): ");
        double p2 = scanner.nextDouble();

        String result = es2.classify(new double[]{p1, p2});
        System.out.println("\n[Результат] Класифікація системою ЕС-2: " + result);
        System.out.println("Висновок: Експертна система стабільна, зручна у діалозі та придатна для практичного використання.");
    }
}