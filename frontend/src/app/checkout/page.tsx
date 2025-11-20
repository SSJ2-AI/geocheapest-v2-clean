'use client'

import { useState, useEffect } from 'react'
import { useCartStore } from '@/store/cartStore'
import { useRouter } from 'next/navigation'
import axios from 'axios'
import { loadStripe } from '@stripe/stripe-js'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

export default function CheckoutPage() {
  const router = useRouter()
  const { items, clearCart } = useCartStore()
  const [optimizedCart, setOptimizedCart] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [shippingAddress, setShippingAddress] = useState({
    name: '',
    street: '',
    city: '',
    province: 'ON',
    postal_code: '',
    country: 'CA',
  })
  const [email, setEmail] = useState('')
  const [userId, setUserId] = useState('')
  const [savePaymentMethod, setSavePaymentMethod] = useState(false)

  useEffect(() => {
    if (items.length === 0) {
      router.push('/cart')
    }
  }, [items, router])

  const handleOptimizeCart = async () => {
    if (!shippingAddress.city || !shippingAddress.postal_code) {
      alert('Please enter your shipping address')
      return
    }

    try {
      setLoading(true)
      const response = await axios.post(`${API_URL}/api/cart/optimize`, {
        items: items.map((item) => ({
          product_id: item.product_id,
          quantity: item.quantity,
        })),
        shipping_address: shippingAddress,
      })
      setOptimizedCart(response.data)
    } catch (error) {
      console.error('Cart optimization failed:', error)
      alert('Failed to optimize cart. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleCheckout = async () => {
    if (!email) {
      alert('Please enter your email')
      return
    }

    if (!optimizedCart) {
      await handleOptimizeCart()
      return
    }

    try {
      setLoading(true)
      const response = await axios.post(`${API_URL}/api/checkout`, {
        items: optimizedCart.items,
        customer_email: email,
        shipping_address: shippingAddress,
        user_id: userId || null,
        save_payment_method: Boolean(userId && savePaymentMethod),
      })

      // Handle Shopify checkout
      if (response.data.shopify_checkout) {
        window.location.href = response.data.shopify_checkout.url
      }

      // Handle affiliate redirects
      if (response.data.affiliate_redirects?.length > 0) {
        alert(
          `You have ${response.data.affiliate_redirects.length} items from affiliate stores. You'll be redirected to complete those purchases.`
        )
        response.data.affiliate_redirects.forEach((redirect: any, index: number) => {
          setTimeout(() => {
            window.open(redirect.url, '_blank')
          }, index * 1000)
        })
      }

      clearCart()
    } catch (error) {
      console.error('Checkout failed:', error)
      alert('Checkout failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Checkout</h1>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Shipping Form */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Shipping Information</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Email
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name
                </label>
                <input
                  type="text"
                  value={shippingAddress.name}
                  onChange={(e) =>
                    setShippingAddress({ ...shippingAddress, name: e.target.value })
                  }
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Street Address
                </label>
                <input
                  type="text"
                  value={shippingAddress.street}
                  onChange={(e) =>
                    setShippingAddress({ ...shippingAddress, street: e.target.value })
                  }
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    City
                  </label>
                  <input
                    type="text"
                    value={shippingAddress.city}
                    onChange={(e) =>
                      setShippingAddress({ ...shippingAddress, city: e.target.value })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    required
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Account ID (optional)
                  </label>
                  <input
                    type="text"
                    value={userId}
                    onChange={(e) => {
                      setUserId(e.target.value)
                      if (!e.target.value) {
                        setSavePaymentMethod(false)
                      }
                    }}
                    placeholder="Enter your GeoCheapest user ID"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                  <p className="text-xs text-gray-500 mt-1">
                    Provide your account ID to save cards for faster checkout.
                  </p>
                </div>

                {userId && (
                  <label className="flex items-center space-x-2 text-sm text-gray-700">
                    <input
                      type="checkbox"
                      checked={savePaymentMethod}
                      onChange={(e) => setSavePaymentMethod(e.target.checked)}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span>Save payment method to my account for future orders</span>
                  </label>
                )}

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Province
                  </label>
                  <select
                    value={shippingAddress.province}
                    onChange={(e) =>
                      setShippingAddress({ ...shippingAddress, province: e.target.value })
                    }
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    {provinces.map((prov) => (
                      <option key={prov} value={prov}>
                        {prov}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Postal Code
                </label>
                <input
                  type="text"
                  value={shippingAddress.postal_code}
                  onChange={(e) =>
                    setShippingAddress({ ...shippingAddress, postal_code: e.target.value })
                  }
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="A1A 1A1"
                  required
                />
              </div>

              {!optimizedCart && (
                <button
                  onClick={handleOptimizeCart}
                  disabled={loading}
                  className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold hover:bg-green-700 transition-colors disabled:bg-gray-400"
                >
                  {loading ? 'Optimizing...' : 'Calculate Best Shipping'}
                </button>
              )}
            </div>
          </div>

          {/* Order Summary */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-6">Order Summary</h2>

            {optimizedCart ? (
              <>
                <div className="space-y-4 mb-6">
                  {optimizedCart.items.map((item: any, index: number) => (
                    <div key={index} className="flex justify-between text-sm">
                      <span className="text-gray-600">
                        Item {index + 1} from {item.source_name}
                      </span>
                      <span className="font-medium">${item.total_price.toFixed(2)}</span>
                    </div>
                  ))}
                </div>

                <div className="border-t pt-4 space-y-2 mb-6">
                  <div className="flex justify-between text-gray-600">
                    <span>Product Total</span>
                    <span>${optimizedCart.total_product_price.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between text-gray-600">
                    <span>Shipping</span>
                    <span>${optimizedCart.total_shipping_cost.toFixed(2)}</span>
                  </div>
                  {optimizedCart.savings > 0 && (
                    <div className="flex justify-between text-green-600 font-semibold">
                      <span>You Saved!</span>
                      <span>${optimizedCart.savings.toFixed(2)}</span>
                    </div>
                  )}
                  <div className="flex justify-between text-xl font-bold text-blue-600 border-t pt-2">
                    <span>Total</span>
                    <span>${optimizedCart.total_price.toFixed(2)}</span>
                  </div>
                </div>

                <button
                  onClick={handleCheckout}
                  disabled={loading}
                  className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors shadow-lg disabled:bg-gray-400"
                >
                  {loading ? 'Processing...' : 'Complete Order'}
                </button>
              </>
            ) : (
              <div className="text-center text-gray-500 py-12">
                <p>Enter your shipping address to see optimized pricing</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
