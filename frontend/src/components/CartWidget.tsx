'use client'

import { useCartStore } from '@/store/cartStore'
import { ShoppingCart } from 'lucide-react'
import { useRouter } from 'next/navigation'

export function CartWidget() {
  const router = useRouter()
  const items = useCartStore((state) => state.items)
  const itemCount = items.reduce((sum, item) => sum + item.quantity, 0)

  return (
    <button
      onClick={() => router.push('/cart')}
      className="relative p-2 rounded-lg hover:bg-gray-100 transition-colors"
    >
      <ShoppingCart className="w-6 h-6 text-gray-700" />
      {itemCount > 0 && (
        <span className="absolute -top-1 -right-1 bg-blue-600 text-white text-xs font-bold rounded-full w-5 h-5 flex items-center justify-center">
          {itemCount}
        </span>
      )}
    </button>
  )
}
