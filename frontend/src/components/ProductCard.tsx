'use client'

import { useCartStore } from '@/store/cartStore'
import { ShoppingCart, Tag } from 'lucide-react'

interface Product {
  id: string
  name: string
  description: string
  category: string
  image_url: string
  best_price: number
  source: string
  source_name: string
  in_stock: boolean
  is_preorder: boolean
}

export function ProductCard({ product }: { product: Product }) {
  const addItem = useCartStore((state) => state.addItem)

  const handleAddToCart = () => {
    addItem({
      product_id: product.id,
      name: product.name,
      price: product.best_price,
      quantity: 1,
      image_url: product.image_url,
    })
  }

  return (
    <div className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow duration-300 overflow-hidden group">
      <div className="relative">
        <img
          src={product.image_url || '/placeholder-product.jpg'}
          alt={product.name}
          className="w-full h-64 object-cover group-hover:scale-105 transition-transform duration-300"
        />
        {product.is_preorder && (
          <span className="absolute top-2 left-2 bg-yellow-500 text-white px-3 py-1 rounded-full text-sm font-semibold">
            Pre-Order
          </span>
        )}
        <span className="absolute top-2 right-2 bg-blue-600 text-white px-3 py-1 rounded-full text-xs font-medium">
          {product.category}
        </span>
      </div>

      <div className="p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2 h-14">
          {product.name}
        </h3>

        <div className="flex items-center gap-2 mb-3">
          <Tag className="w-4 h-4 text-gray-500" />
          <span className="text-sm text-gray-600">{product.source_name}</span>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <span className="text-2xl font-bold text-blue-600">
              ${product.best_price.toFixed(2)}
            </span>
            <span className="text-sm text-gray-500 ml-1">CAD</span>
          </div>

          <button
            onClick={handleAddToCart}
            disabled={!product.in_stock && !product.is_preorder}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              product.in_stock || product.is_preorder
                ? 'bg-blue-600 text-white hover:bg-blue-700'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
            }`}
          >
            <ShoppingCart className="w-4 h-4" />
            Add
          </button>
        </div>

        {!product.in_stock && !product.is_preorder && (
          <p className="text-red-500 text-sm mt-2 font-medium">Out of Stock</p>
        )}
      </div>
    </div>
  )
}
