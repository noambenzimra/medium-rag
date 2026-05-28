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

async function test() {
  console.log('🧪 TEST 1: Embedding endpoint');
  console.log('─────────────────────────────');

  const res = await openai.embeddings.create({
    model: '4UHRUIN-text-embedding-3-small',
    input: 'This is a simple test sentence.',
  });

  const embedding = res.data[0].embedding;

  console.log('Type of embedding:', typeof embedding);
  console.log('Is array?:', Array.isArray(embedding));
  console.log('Length:', embedding?.length);
  console.log('First 3 values:', embedding?.slice(0, 3));
  console.log('Type of first value:', typeof embedding?.[0]);
  console.log('');

  console.log('🧪 TEST 2: Single Pinecone upsert');
  console.log('─────────────────────────────');

  const testVector = {
    id: 'test-vector-1',
    values: embedding,
    metadata: { title: 'Test', chunk: 'hello world' },
  };

  console.log('Upserting 1 test vector...');
  await index.upsert({ records: [testVector] });
  console.log('✅ SUCCESS! Pinecone upsert works.');
}

test().catch((err) => {
  console.error('❌ ERROR:', err.message);
  console.error(err);
});