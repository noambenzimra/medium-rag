import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

// The question we want to test
const QUESTION = 'an article about building habits that actually stick';
const TOP_K = 5;

async function queryTest() {
  console.log(`❓ Question: "${QUESTION}"\n`);

  // 1. Turn the question into a vector
  console.log('🔢 Embedding the question...');
  const embRes = await openai.embeddings.create({
    model: '4UHRUIN-text-embedding-3-small',
    input: QUESTION,
  });
  const queryVector = embRes.data[0].embedding;

  // 2. Ask Pinecone for the most similar chunks
  console.log(`🔍 Searching Pinecone for top ${TOP_K} matches...\n`);
  const results = await index.query({
    vector: queryVector,
    topK: TOP_K,
    includeMetadata: true,
  });

  // 3. Show what came back
  console.log('📊 RESULTS:');
  console.log('═══════════════════════════════════════\n');

  results.matches.forEach((match, i) => {
    console.log(`#${i + 1}  (score: ${match.score.toFixed(4)})`);
    console.log(`    Title:  ${match.metadata.title}`);
    console.log(`    Author: ${match.metadata.authors}`);
    console.log(`    Chunk:  ${String(match.metadata.chunk).slice(0, 150)}...`);
    console.log('');
  });

  console.log('✅ Query test complete!');
  console.log('💡 Higher score = more relevant. Cosine scores range roughly 0 to 1.');
}

queryTest().catch(console.error);