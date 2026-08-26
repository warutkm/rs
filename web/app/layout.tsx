import type { Metadata } from 'next';
import { Inter, Outfit } from 'next/font/google';
import './globals.css';
import { UserProvider } from '@/context/UserContext';
import Navbar from '@/components/Navbar';
import Link from 'next/link';
import { Sparkles, Database, Server, GitFork, Cpu } from 'lucide-react';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const outfit = Outfit({ subsets: ['latin'], variable: '--font-outfit' });

export const metadata: Metadata = {
  title: 'Amazon RecSys v2 | Two-Stage Retrieval & LambdaMART Ranking',
  description: 'Production-grade two-stage recommendation platform with LightGBM LambdaMART, Qdrant ANN, Two-Tower embeddings, and Gemini 3.5 Flash-Lite LLM explanations.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${outfit.variable} font-sans min-h-screen flex flex-col bg-slate-950 text-slate-100`}>
        <UserProvider>
          <Navbar />
          <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
            {children}
          </main>
          
          {/* Footer */}
          <footer className="border-t border-slate-900 bg-slate-950/80 py-8 mt-12 text-xs text-slate-500">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 rounded bg-sky-500/20 text-sky-400 flex items-center justify-center font-bold text-[10px]">
                  v2
                </div>
                <span>Amazon RecSys v2 Production Platform · DS11 Rework</span>
              </div>

              <div className="flex items-center gap-6 text-slate-400">
                <span className="flex items-center gap-1">
                  <Database className="w-3.5 h-3.5 text-sky-400" />
                  Qdrant + LightGBM
                </span>
                <span className="flex items-center gap-1">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                  Gemini 3.5 Flash-Lite
                </span>
                <span className="flex items-center gap-1">
                  <Server className="w-3.5 h-3.5 text-emerald-400" />
                  FastAPI Async
                </span>
                <Link href="/admin" className="hover:text-amber-400 transition-colors">
                  Telemetry & Retrain
                </Link>
              </div>
            </div>
          </footer>
        </UserProvider>
      </body>
    </html>
  );
}
