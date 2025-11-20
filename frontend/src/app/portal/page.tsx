'use client'

import Link from 'next/link'
import { Shield, Store, Users } from 'lucide-react'

const cards = [
  {
    title: 'Customer / Registered User',
    description: 'View your orders, saved cards, and checkout faster.',
    href: '/user/dashboard',
    icon: Users,
  },
  {
    title: 'Vendor Portal',
    description: 'Sync Shopify products, monitor payouts, and print Shippo labels.',
    href: '/vendor/dashboard',
    icon: Store,
  },
  {
    title: 'Super Admin',
    description: 'Approve vendors, monitor platform health, and tweak commission.',
    href: '/admin/dashboard',
    icon: Shield,
  },
]

export default function PortalPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-10">
          <h1 className="text-4xl font-bold text-gray-900">Choose Your Portal</h1>
          <p className="text-gray-600 mt-3">
            Access the experience tailored for customers, vendors, or super admins.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {cards.map((card) => (
            <Link
              key={card.title}
              href={card.href}
              className="bg-white rounded-2xl shadow-md hover:shadow-lg transition-shadow p-6 flex flex-col gap-4 border border-gray-100"
            >
              <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
                <card.icon className="w-6 h-6" />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-gray-900">{card.title}</h2>
                <p className="text-gray-600 mt-2">{card.description}</p>
              </div>
              <span className="text-blue-600 font-medium">Enter portal →</span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
