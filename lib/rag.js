import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';

// ─── CLIENTS ───────────────────────────────────────
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX_NAME);
// ──────────────────────────────────────────────────

// ─── CONFIG (must match what we used to ingest) ────
export const CONFIG = {
  chunk_size: 512,
  overlap_ratio: 0.2,
  top_k: 5,
};

// How many candidate chunks to fetch BEFORE de-duplicating.
// We over-fetch so that after keeping one chunk per article,
// we still have enough distinct articles to fill top_k.
const FETCH_MULTIPLIER = 5;
// ──────────────────────────────────────────────────

// The REQUIRED system prompt from the assignment.
// Do NOT remove the constraints — only safe to add style notes.
const SYSTEM_PROMPT = `You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data." Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful.

Response style: be concise and direct. When asked for a title and author, state them clearly. When asked for a list, return only what was requested.`;

// STEP 1: Turn the user's question into a vector
async function embedQuestion(question) {
  const res = await openai.embeddings.create({
    model: '4UHRUIN-text-embedding-3-small',
    input: question,
  });
  return res.data[0].embedding;
}

// STEP 2: Search Pinecone for candidate chunks (over-fetch)
async function retrieve(queryVector, candidateCount) {
  const results = await index.query({
    vector: queryVector,
    topK: candidateCount,
    includeMetadata: true,
  });
  return results.matches || [];
}

// STEP 2.5: Keep only the best-scoring chunk per unique article,
// then return the top `limit` distinct articles.
function dedupeByArticle(matches, limit) {
  const bestPerArticle = new Map();

  for (const m of matches) {
    const articleId = m.metadata.article_id;
    const existing = bestPerArticle.get(articleId);
    // Pinecone returns matches sorted by score (highest first),
    // so the first time we see an article it's already the best.
    if (!existing) {
      bestPerArticle.set(articleId, m);
    }
  }

  // Convert back to an array, keep original score order, take top `limit`
  return Array.from(bestPerArticle.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
}

// STEP 3: Build the context block + user prompt from retrieved chunks
function buildUserPrompt(question, matches) {
  const contextBlock = matches
    .map((m, i) => {
      return `[Source ${i + 1}]
Title: ${m.metadata.title}
Author: ${m.metadata.authors}
Tags: ${m.metadata.tags}
Passage: ${m.metadata.chunk}`;
    })
    .join('\n\n---\n\n');

  const userPrompt = `Here is the retrieved context from the Medium articles dataset:

${contextBlock}

Based ONLY on the context above, answer this question:
${question}`;

  return userPrompt;
}

// STEP 4: Ask gpt-5-mini to answer using only the context
async function generateAnswer(systemPrompt, userPrompt) {
  const res = await openai.chat.completions.create({
    model: '4UHRUIN-gpt-5-mini',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
  });
  return res.choices[0].message.content;
}

// ─── THE MAIN FUNCTION — runs the full RAG pipeline ──
export async function runRAG(question) {
  // 1. Embed the question
  const queryVector = await embedQuestion(question);

  // 2. Retrieve a LARGER pool of candidate chunks
  const candidateCount = CONFIG.top_k * FETCH_MULTIPLIER;
  const candidates = await retrieve(queryVector, candidateCount);

  // 2.5. De-duplicate: one chunk per article, then take top_k distinct articles
  const matches = dedupeByArticle(candidates, CONFIG.top_k);

  // 3. Build the augmented prompt
  const userPrompt = buildUserPrompt(question, matches);

  // 4. Generate the answer
  const answer = await generateAnswer(SYSTEM_PROMPT, userPrompt);

  // 5. Shape the context array for the response (per assignment format)
  const context = matches.map((m) => ({
    article_id: m.metadata.article_id,
    title: m.metadata.title,
    chunk: m.metadata.chunk,
    score: m.score,
  }));

  // 6. Return everything the API needs
  return {
    response: answer,
    context: context,
    augmented_prompt: {
      System: SYSTEM_PROMPT,
      User: userPrompt,
    },
  };
}