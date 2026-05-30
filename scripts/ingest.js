import { createReadStream, existsSync, readFileSync, writeFileSync } from 'fs';
import { parse } from 'csv-parse';
import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

// ─── CLIENTS ───────────────────────────────────────
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX_NAME);
// ──────────────────────────────────────────────────

// ─── HYPERPARAMETERS ──────────────────────────────
const CHUNK_SIZE    = 512;
const OVERLAP       = 0.2;
const LIMIT         = 7682;
const PROGRESS_FILE = './ingest-progress.json';
// ──────────────────────────────────────────────────

function loadProgress() {
  if (existsSync(PROGRESS_FILE)) {
    const data = JSON.parse(readFileSync(PROGRESS_FILE, 'utf-8'));
    console.log(`🔄 Resuming from article ${data.lastArticle + 1}...`);
    return data.lastArticle;
  }
  return -1;
}

function saveProgress(i) {
  writeFileSync(PROGRESS_FILE, JSON.stringify({ lastArticle: i }));
}

function chunkText(text) {
  const words = text.split(/\s+/).filter(Boolean);
  const overlapWords = Math.floor(CHUNK_SIZE * OVERLAP);
  const chunks = [];
  let start = 0;
  while (start < words.length) {
    const end = Math.min(start + CHUNK_SIZE, words.length);
    const chunk = words.slice(start, end).join(' ').trim();
    if (chunk) chunks.push(chunk);
    if (end === words.length) break;
    start += CHUNK_SIZE - overlapWords;
  }
  return chunks;
}

async function embed(text) {
  const res = await openai.embeddings.create({
    model: '4UHRUIN-text-embedding-3-small',
    input: text,
  });
  return res.data[0].embedding;
}

async function ingest() {
  const lastArticle = loadProgress();

  console.log('📖 Reading CSV...');
  const articles = [];

  await new Promise((resolve, reject) => {
    createReadStream('./medium-english-50mb.csv')
      .pipe(parse({ columns: true, skip_empty_lines: true }))
      .on('data', (row) => {
        if (articles.length < LIMIT) articles.push(row);
      })
      .on('end', resolve)
      .on('error', reject);
  });

  console.log(`✅ Loaded ${articles.length} articles`);

  // Debug: show columns of first article
  if (articles.length > 0) {
    console.log(`🔍 CSV columns found: ${Object.keys(articles[0]).join(', ')}`);
    console.log(`🔍 First article text preview: "${String(articles[0].text || '').slice(0, 80)}..."`);
  }

  console.log('🔪 Chunking and embedding...\n');

  for (let i = 0; i < articles.length; i++) {
    if (i <= lastArticle) {
      console.log(`⏭️  Skipping article ${i + 1} (already done)`);
      continue;
    }

    const article = articles[i];
    const text = String(article.text || '').trim();

    if (!text) {
      console.log(`⚠️  Article ${i + 1} is empty, skipping`);
      saveProgress(i);
      continue;
    }

    const chunks = chunkText(text);
    console.log(`📄 Article ${i + 1}: "${article.title?.slice(0, 50)}" → ${chunks.length} chunks`);

    if (chunks.length === 0) {
      console.log(`⚠️  Article ${i + 1} produced 0 chunks, skipping`);
      saveProgress(i);
      continue;
    }

    const vectors = [];

    for (let j = 0; j < chunks.length; j++) {
      const embedding = await embed(chunks[j]);
      vectors.push({
        id: `art${i}-ch${j}`,
        values: embedding,
        metadata: {
          article_id: String(i),
          title: String(article.title || ''),
          authors: String(article.authors || ''),
          tags: String(article.tags || ''),
          chunk: chunks[j],
        },
      });
    }

console.log(`   📦 Uploading ${vectors.length} vectors...`);
    console.log(`   🔍 First vector check:`, {
      id: vectors[0]?.id,
      valuesType: typeof vectors[0]?.values,
      valuesIsArray: Array.isArray(vectors[0]?.values),
      valuesLength: vectors[0]?.values?.length,
    });
    await index.upsert({ records: vectors });
    console.log(`   ✅ Uploaded!\n`);

    saveProgress(i);
    console.log(`📝 Article ${i + 1}/${articles.length} complete\n`);
  }

  console.log('🎉 Ingestion complete!');
}

ingest().catch(console.error);