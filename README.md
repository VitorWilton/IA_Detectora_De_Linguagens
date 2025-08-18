Com certeza. Faz sentido atualizar a documentação do projeto para refletir as mudanças que fizemos.

Aqui está o README.md revisado, que agora considera os novos arquivos de treino train.jsonl (Java) e python_train.jsonl que geramos.

Detector de Linguagens de Programação com IA
Este projeto é uma aplicação Java que implementa um agente de inteligência artificial para classificar a linguagem de programação de um dado trecho de código. Ele expõe uma API RESTful que, ao receber um trecho de código, retorna se a linguagem é Java ou Python.

A IA foi treinada com um grande volume de dados de código de ambas as linguagens para garantir alta precisão nas previsões.

Funcionalidades Principais
API RESTful: Comunicação via endpoint HTTP (/classify).

Classificação de Código: O modelo de IA identifica a linguagem do trecho de código.

Treinamento com Dados Reais: O modelo é treinado em tempo de execução com grandes datasets de código em formatos JSON e CSV.

Tecnologias de IA em Java: Utiliza as bibliotecas Deeplearning4j e Apache Commons CSV para o processamento e treinamento do modelo.

Tecnologias Utilizadas
Java 21

Spring Boot 3

Apache Maven

Deeplearning4j (DL4J)

Apache Commons CSV

Jackson (para processamento de JSON)

Como Rodar o Projeto
Pré-requisitos:

Java Development Kit (JDK) 21 ou superior

Apache Maven

Clone o repositório:
git clone [URL_DO_SEU_REPOSITORIO]

Verifique os Datasets de Treino:
Os arquivos de treino (train.jsonl para Java e python_train.jsonl para Python) já estão na pasta src/main/resources/data.

Execute a aplicação:
No terminal, navegue até a pasta raiz do projeto e execute o comando:
./mvnw spring-boot:run

A aplicação iniciará na porta 8080.

Como Usar a API
Com a aplicação em execução, você pode enviar requisições POST para o endpoint /classify.

Exemplo de requisição com código Java:

Bash

curl -X POST -H "Content-Type: application/json" -d "{\"code\":\"public static void main(String[] args) {}\"}" http://localhost:8080/classify
Exemplo de requisição com código Python:

Bash

curl -X POST -H "Content-Type: application/json" -d "{\"code\":\"def hello_world(): pass\"