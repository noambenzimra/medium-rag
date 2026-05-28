import { NextResponse } from 'next/server';
import { CONFIG } from '../../../lib/rag.js';

// GET /api/stats
export async function GET() {
  // Return the current RAG configuration.
  // Field names MUST match the assignment exactly.
  return NextResponse.json({
    chunk_size: CONFIG.chunk_size,
    overlap_ratio: CONFIG.overlap_ratio,
    top_k: CONFIG.top_k,
  });
}