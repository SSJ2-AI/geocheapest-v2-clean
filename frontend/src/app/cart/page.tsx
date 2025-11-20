'use client'

import { useCartStore } from '@/store/cartStore'
import { useRouter } from 'next/navigation'
import { Trash2, Plus, Minus } from 'lucide-react'

export default function CartPage() {
  const router = useRouter()
  const { items, removeItem, updateQuantity, getTotalPrice } = useCartStore()

  const handleCheckout = () => {
    router.push('/checkout')
  }

  if (items.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Your cart is empty</h2>
          <p className="text-gray-600 mb-8">Add some TCG products to get started!</p>
          <button
            onClick={() => router.push('/')}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Continue Shopping
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <button
            onClick={() => router.push('/')}
            className="text-blue-600 hover:text-blue-700 font-medium"
          >
            ← Continue Shopping
          </button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Cart Items */}
          <div className="lg:col-span-2 space-y-4">
            <h1 className="text-3xl font-bold text-gray-900 mb-6">Shopping Cart</h1>

            {items.map((item) => (
              <div
                key={item.product_id}
                className="bg-white rounded-lg shadow-md p-6 flex gap-4"
              >
                <img
                  src={item.image_url || '/placeholder-product.jpg'}
                  alt={item.name}
                  className="w-24 h-24 object-cover rounded-md"
                />

                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 mb-2">{item.name}</h3>
                  <p className="text-blue-600 font-bold">
                    ${item.price.toFixed(2)} CAD
                  </p>

                  <div className="flex items-center gap-3 mt-4">
                    <button
                      onClick={() => updateQuantity(item.product_id, Math.max(1, item.quantity - 1))}
                      className="p-1 rounded-md border border-gray-300 hover:bg-gray-100"
                    >
                      <Minus className="w-4 h-4" />
                    </button>
                    <span className="w-12 text-center font-medium">{item.quantity}</span>
                    <button
                      onClick={() => updateQuantity(item.product_id, item.quantity + 1)}
                      className="p-1 rounded-md border border-gray-300 hover:bg-gray-100"
                    >
                      <Plus className="w-4 h-4" />
                    </button>

                    <button
                      onClick={() => removeItem(item.product_id)}
                      className="ml-auto text-red-600 hover:text-red-700"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>

                <div className="text-right">
                  <p className="text-gray-500 text-sm mb-1">Subtotal</p>
                  <p className="text-xl font-bold text-gray-900">
                    ${(item.price * item.quantity).toFixed(2)}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Order Summary */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6 sticky top-8">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Order Summary</h2>

              <div className="space-y-3 mb-6">
                <div className="flex justify-between text-gray-600">
                  <span>Subtotal ({items.length} items)</span>
                  <span>${getTotalPrice().toFixed(2)}</span>
                </div>
                <div className="flex justify-between text-gray-600">
                  <span>Shipping</span>
                  <span className="text-green-600 font-medium">Calculated at checkout</span>
                </div>
                <div className="border-t pt-3 flex justify-between text-lg font-bold">
                  <span>Estimated Total</span>
                  <span className="text-blue-600">${getTotalPrice().toFixed(2)}</span>
                </div>
              </div>

              <button
                onClick={handleCheckout}
                className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors shadow-lg"
              >
                Proceed to Checkout
              </button>

              <p className="text-xs text-gray-500 text-center mt-4">
                We'll find you the cheapest shipping at checkout!
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
