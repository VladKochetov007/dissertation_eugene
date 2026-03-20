import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dissertation Knowledge Graph",
  description: "Interactive knowledge graph for LSE Mathematics Dissertation",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
