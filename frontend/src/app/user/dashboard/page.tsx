'use client'

import { useState } from 'react'
import axios from 'axios'
import { User, CreditCard, Package, DollarSign, Search } from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export default function UserDashboard() {
  const [userId, setUserId] = useState('')
  const [profile, setProfile] = useState<any>(null)
  const [orders, setOrders] = useState<any[]>([])
  const [paymentMethods, setPaymentMethods] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const handleLoadProfile = async () => {
    if (!userId) {
      alert('Enter your user ID (sent via order confirmation).')
      return
    }

    try {
      setLoading(true)
      const [profileRes, ordersRes, paymentsRes] = await Promise.all([
        axios.get(`${API_URL}/api/users/${userId}`),
        axios.get(`${API_URL}/api/users/${userId}/orders`),
        axios.get(`${API_URL}/api/users/${userId}/payment-methods`),
      ])
      setProfile(profileRes.data)
      setOrders(ordersRes.data.orders || [])
      setPaymentMethods(paymentsRes.data.payment_methods || [])
    } catch (error) {
      console.error('Failed to fetch user data', error)
      alert('Unable to load your dashboard. Double-check the user ID.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-10">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">User Portal</h1>
            <p className="text-gray-600 mt-1">
              View your saved payment methods and order history
            </p>
          </div>
          <div className="flex items-center gap-3">
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="Enter user ID"
              className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            <button
              onClick={handleLoadProfile}
              disabled={loading}
              className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400"
            >
              <Search className="w-4 h-4" />
              {loading ? 'Loading...' : 'Load'}
            </button>
          </div>
        </div>

        {profile ? (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center gap-4">
                  <div className="bg-blue-100 p-3 rounded-lg">
                    <User className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Stripe Customer</p>
                    <p className="text-sm font-mono text-gray-900">
                      {profile.user?.stripe_customer_id || 'Not linked'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center gap-4">
                  <div className="bg-green-100 p-3 rounded-lg">
                    <Package className="w-6 h-6 text-green-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Orders</p>
                    <p className="text-2xl font-bold text-gray-900">
                      {profile.stats?.orders || 0}
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center gap-4">
                  <div className="bg-purple-100 p-3 rounded-lg">
                    <DollarSign className="w-6 h-6 text-purple-600" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Lifetime Spend</p>
                    <p className="text-2xl font-bold text-gray-900">
                      ${profile.stats?.lifetime_value?.toFixed(2) || '0.00'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-gray-900">Payment Methods</h2>
                  <CreditCard className="w-5 h-5 text-blue-600" />
                </div>
                {paymentMethods.length > 0 ? (
                  <div className="space-y-4">
                    {paymentMethods.map((card) => (
                      <div
                        key={card.id}
                        className="flex items-center justify-between border rounded-lg p-4"
                      >
                        <div>
                          <p className="font-medium text-gray-900">
                            {card.brand} ending in {card.last4}
                          </p>
                          <p className="text-sm text-gray-600">
                            Expires {card.exp_month}/{card.exp_year}
                          </p>
                        </div>
                        <span className="text-sm text-gray-500 uppercase">
                          {card.funding}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-8">
                    No saved payment methods yet.
                  </p>
                )}
              </div>

              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-gray-900">Recent Orders</h2>
                  <Package className="w-5 h-5 text-green-600" />
                </div>
                {orders.length > 0 ? (
                  <div className="space-y-4 max-h-[420px] overflow-y-auto">
                    {orders.map((order) => (
                      <div key={order.id} className="border rounded-lg p-4">
                        <div className="flex justify-between">
                          <span className="font-semibold text-gray-800">Order #{order.id}</span>
                          <span className="text-blue-600 font-bold">
                            ${order.total_amount?.toFixed(2) || '0.00'}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">
                          {order.status?.toUpperCase()}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">
                          {order.created_at
                            ? new Date(order.created_at).toLocaleString()
                            : ''}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-8">No orders yet.</p>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="bg-white rounded-lg shadow-md p-12 text-center text-gray-500">
            Enter your user ID to load your dashboard.
          </div>
        )}
      </div>
    </div>
  )
}
