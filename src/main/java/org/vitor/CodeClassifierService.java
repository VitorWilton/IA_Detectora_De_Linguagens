package org.vitor.ia;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;
import jakarta.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class CodeClassifierService {

    private MultiLayerNetwork model;
    private Map<String, Integer> wordToIndex = new HashMap<>();
    private final List<String> labels = Arrays.asList("Java", "Python");
    private int vocabSize;

    @PostConstruct
    public void init() {
        try {
            trainModel();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void trainModel() throws IOException {
        List<String> allCodeSnippets = new ArrayList<>();
        List<Integer> allLabels = new ArrayList<>();

        // -- Carregar o dataset de Python (JSONL) --
        String pythonFilePath = "data/python_train.jsonl";
        ObjectMapper objectMapper = new ObjectMapper();

        try (InputStream is = getClass().getClassLoader().getResourceAsStream(pythonFilePath);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (allCodeSnippets.size() >= 10000) break;
                JsonNode jsonNode = objectMapper.readTree(line);
                String code = jsonNode.get("text").asText();
                allCodeSnippets.add(code);
                allLabels.add(1); // 1 = Python
            }
        }

        // -- Carregar o novo dataset de Java (JSONL) --
        String javaFilePath = "data/java_train.jsonl";
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(javaFilePath);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (allCodeSnippets.size() >= 10000) break;
                JsonNode jsonNode = objectMapper.readTree(line);
                // ATENÇÃO: A chave no JSON para o código pode ser diferente de 'text'
                String code = jsonNode.get("text").asText();
                allCodeSnippets.add(code);
                allLabels.add(0); // 0 = Java
            }
        }

        int index = 0;
        for (String snippet : allCodeSnippets) {
            String[] tokens = snippet.toLowerCase().split("\\W+");
            for (String token : tokens) {
                if (!token.isEmpty() && !wordToIndex.containsKey(token)) {
                    wordToIndex.put(token, index++);
                }
            }
        }
        this.vocabSize = wordToIndex.size();

        INDArray features = Nd4j.zeros(allCodeSnippets.size(), vocabSize);
        INDArray outputLabels = Nd4j.zeros(allCodeSnippets.size(), 2);

        for (int i = 0; i < allCodeSnippets.size(); i++) {
            String snippet = allCodeSnippets.get(i);
            String[] tokens = snippet.toLowerCase().split("\\W+");
            for (String token : tokens) {
                if (wordToIndex.containsKey(token)) {
                    features.putScalar(i, wordToIndex.get(token), 1.0);
                }
            }
            outputLabels.putScalar(i, allLabels.get(i), 1.0);
        }
        DataSet dataSet = new DataSet(features, outputLabels);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Nesterovs(0.01))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(vocabSize).nOut(10)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(10).nOut(2)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.fit(dataSet);
    }

    public String classifyCode(String codeSnippet) {
        INDArray testFeature = Nd4j.zeros(1, vocabSize);
        String[] tokens = codeSnippet.toLowerCase().split("\\W+");
        for (String token : tokens) {
            if (wordToIndex.containsKey(token)) {
                testFeature.putScalar(0, wordToIndex.get(token), 1.0);
            }
        }

        INDArray prediction = model.output(testFeature);
        int predictedClass = Nd4j.argMax(prediction, 1).getInt(0);
        return labels.get(predictedClass);
    }
}