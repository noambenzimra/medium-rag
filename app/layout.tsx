export const metadata = {
  title: 'Medium RAG',
  description: 'RAG over Medium articles',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}