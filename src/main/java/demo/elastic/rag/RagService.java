package demo.elastic.rag;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.elasticsearch.ElasticsearchVectorStore;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class RagService {

    // Both beans autowired from default configuration
    private ElasticsearchVectorStore vectorStore;
    private ChatClient chatClient;

    public RagService(ElasticsearchVectorStore vectorStore, ChatClient.Builder clientBuilder) {
        this.vectorStore = vectorStore;
        this.chatClient = clientBuilder.build();
    }

    public void ingestPDF(String path) {

        // Spring AI utility class to read a PDF file page by page
        PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(path);
        List<Document> docbatch = pdfReader.read();

        // Sending batch of documents to vector store
        // applying tokenizer
        docbatch = new TokenTextSplitter().apply(docbatch);
        vectorStore.doAdd(docbatch);
    }

    public String queryLLM(String question) {

        // Querying the vector store for documents related to the question
        List<Document> vectorStoreResult =
            vectorStore.doSimilaritySearch(SearchRequest.builder().query(question).topK(5)
                    .similarityThreshold(0.6).build());

        // Merging the documents into a single string
        String documents = vectorStoreResult.stream()
            .map(Document::getText)
            .collect(Collectors.joining(System.lineSeparator()));

        // Setting the prompt with the context
        String prompt = """
            You're assisting with providing the rules of the tabletop game Runewars.
            Use the information from the DOCUMENTS section to provide accurate answers to the
            question in the QUESTION section. 
            If unsure, simply state that you don't know.
            
            DOCUMENTS:
            """ + documents
            + """
            QUESTION:
            """ + question;


        // Calling the chat model with the question
        String response = chatClient.prompt()
            .user(prompt)
            .call()
            .content();

        return response +
            System.lineSeparator() +
            "Found at page: " +
            // Retrieving the first ranked page number from the document metadata
            vectorStoreResult.get(0).getMetadata().get(PagePdfDocumentReader.METADATA_START_PAGE_NUMBER) +
            " of the manual";
    }
}
