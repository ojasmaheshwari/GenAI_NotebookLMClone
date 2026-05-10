import "dotenv/config";
import express from "express";
import multer from "multer";
import path from "path";

import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf";

import {RecursiveCharacterTextSplitter} from "langchain/text_splitter";

import {OpenAIEmbeddings, ChatOpenAI} from "@langchain/openai";

import {QdrantVectorStore} from "@langchain/qdrant";

import cors from "cors";

const app = express();

app.use(express.json());
app.use(cors());

const PORT = 3000;

// ==============================
// Multer Config
// ==============================

const upload = multer({
	storage: multer.memoryStorage(),
});

// ==============================
// OpenRouter Config
// ==============================

const llm = new ChatOpenAI({
	model: "openai/gpt-4.1-mini",
	temperature: 0,
	maxTokens: 500,

	apiKey: process.env.OPENROUTER_API_KEY,

	configuration: {
		baseURL: "https://openrouter.ai/api/v1",
	},
});

const embeddings = new OpenAIEmbeddings({
	model: "text-embedding-3-small",

	apiKey: process.env.OPENROUTER_API_KEY,

	configuration: {
		baseURL: "https://openrouter.ai/api/v1",
	},
});

// ==============================
// Upload + Index Route
// ==============================

app.post("/upload", upload.single("file"), async (req, res) => {
	try {
		if (!req.file) {
			return res.status(400).json({
				error: "No file uploaded",
			});
		}

		const originalName = req.file.originalname;

		let docs;

		// ==============================
		// Load File
		// ==============================

		if (originalName.endsWith(".pdf")) {
			const blob = new Blob([req.file.buffer], {
				type: "application/pdf",
			});

			const loader = new PDFLoader(blob);

			docs = await loader.load();
		} else if (originalName.endsWith(".txt")) {
			const text = req.file.buffer.toString("utf-8");

			docs = [
				{
					pageContent: text,
					metadata: {
						source: originalName,
					},
				},
			];
		} else {
			return res.status(400).json({
				error: "Only PDF and TXT files are supported",
			});
		}

		// ==============================
		// Chunking
		// ==============================

		const splitter = new RecursiveCharacterTextSplitter({
			chunkSize: 1000,
			chunkOverlap: 200,
		});

		const splitDocs = await splitter.splitDocuments(docs);

		// ==============================
		// Vector DB
		// ==============================

		const collectionName =
			path.parse(originalName).name.toLowerCase() +
			"-" +
			Date.now();

		await QdrantVectorStore.fromDocuments(splitDocs, embeddings, {
			url: process.env.QDRANT_URL,
			apiKey: process.env.QDRANT_API_KEY,
			collectionName,
		});

		res.json({
			success: true,
			message: "Document indexed successfully",
			collectionName,
			chunks: splitDocs.length,
		});
	} catch (err) {
		console.error(err);

		res.status(500).json({
			error: "Internal server error",
		});
	}
});

// ==============================
// Ask Question Route
// ==============================

app.post("/ask", async (req, res) => {
	try {
		const {question, collectionName} = req.body;

		if (!question || !collectionName) {
			return res.status(400).json({
				error: "question and collectionName are required",
			});
		}

		// ==============================
		// Load Existing Collection
		// ==============================

		const vectorStore =
			await QdrantVectorStore.fromExistingCollection(
				embeddings,
				{
					url: process.env.QDRANT_URL,
					apiKey: process.env.QDRANT_API_KEY,
					collectionName,
				}
			);

		// ==============================
		// Retrieval
		// ==============================

		const retriever = vectorStore.asRetriever({
			k: 4,
		});

		const retrievedDocs = await retriever.invoke(question);

		// ==============================
		// Context Formatting
		// ==============================

		const context = retrievedDocs
			.map((doc, index) => {
				return `
Chunk ${index + 1}
Page: ${doc.metadata?.loc?.pageNumber || "Unknown"}

${doc.pageContent}
`;
			})
			.join("\n\n");

		// ==============================
		// Prompt
		// ==============================

		const prompt = `
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context, say:
"I could not find the answer in the uploaded document."

Context:
${context}

Question:
${question}
`;

		// ==============================
		// LLM Call
		// ==============================

		const response = await llm.invoke(prompt);

		// ==============================
		// Return Response
		// ==============================

		res.json({
			answer: response.content,
			sources: retrievedDocs.map((doc) => ({
				page: doc.metadata?.loc?.pageNumber || "Unknown",
				preview: doc.pageContent.slice(0, 200),
			})),
		});
	} catch (err) {
		console.error(err);

		res.status(500).json({
			error: "Internal server error",
		});
	}
});

app.listen(PORT, () => {
	console.log(`Server running on http://localhost:${PORT}`);
});
