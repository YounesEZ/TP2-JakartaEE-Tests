package ma.emsi;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiEmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.store.embedding.CosineSimilarity;

import java.time.Duration;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        ChatLanguageModel model = GoogleAiGeminiChatModel.builder()
                .temperature(0.7)
                .modelName("gemini-1.5-flash")
                .apiKey(System.getenv("GEMINI_KEY"))
                .build();
//     //Test 1
//        String message = "Explain GenAI";
//
//        String stringResponce = model.generate(message);
//
//        System.out.println(stringResponce);

        // Test 2

        String texteATraduire = "Boujour, comment allez vous.";

        Prompt prompt = PromptTemplate
                .from("Traduire en {{langue}} ce texte : {{text}}")
                .apply(Map.of(
                        "langue", "anglais",
                        "text", texteATraduire
                ));

        String responce = model.generate(prompt.text());
        System.out.println(responce);

    }
}