package org.vitor;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import java.util.Map;

@RestController
public class CodeClassifierController {

    private final org.vitor.CodeClassifierService classifierService;

    @Autowired
    public CodeClassifierController(org.vitor.CodeClassifierService classifierService) {
        this.classifierService = classifierService;
    }

    @PostMapping("/classify")
    public String classifyCode(@RequestBody Map<String, String> requestBody) {
        String codeSnippet = requestBody.get("code");
        if (codeSnippet == null || codeSnippet.isEmpty()) {
            return "Por favor, forneça um trecho de código.";
        }
        String result = classifierService.classifyCode(codeSnippet);
        return "A IA previu que a linguagem é: " + result;
    }
}