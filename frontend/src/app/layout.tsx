import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import ChatWindow from '../../components/ChatWindow'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'GeoCheapest - Best TCG Prices in Canada',
  description: 'Find the cheapest Pokemon, Yu-Gi-Oh, Magic, and One Piece cards in Canada',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {children}
        <ChatWindow />
      </body>
    </html>
  )
}
