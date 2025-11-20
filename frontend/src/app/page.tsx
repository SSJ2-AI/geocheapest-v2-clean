'use client'

import { useEffect, useState } from 'react'
import { ProductCard } from '@/components/ProductCard'
import { CartWidget } from '@/components/CartWidget'
import { SearchBar } from '@/components/SearchBar'
import { useCartStore } from '@/store/cartStore'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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

export default function Home() {
  const [products, setProducts] = useState<Product[]>([])
  const [loading, setLoading] = useState(true)
  const [category, setCategory] = useState<string | null>(null)

  useEffect(() => {
    fetchProducts()
  }, [category])

  const fetchProducts = async () => {
    try {
      setLoading(true)
      const params = category ? { category } : {}
      const response = await axios.get(`${API_URL}/api/products`, { params })
      setProducts(response.data.products)
    } catch (error) {
      console.error('Failed to fetch products:', error)
    } finally {
      setLoading(false)
    }
  }

  const categories = ['Pokemon', 'One Piece', 'Yu-Gi-Oh', 'Magic: The Gathering', 'Dragon Ball']

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 text-transparent bg-clip-text">
                GeoCheapest
              </h1>
              <span className="ml-2 text-sm text-gray-500">Canada's TCG Marketplace</span>
            </div>
            <CartWidget />
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="bg-gradient-to-r from-blue-600 to-purple-600 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-4xl font-bold mb-4">
            Find the Best TCG Prices in Canada
          </h2>
          <p className="text-xl mb-8 text-blue-100">
            We compare prices from multiple vendors to save you money
          </p>
          <SearchBar onSearch={(query) => console.log('Search:', query)} />
        </div>
      </section>

      {/* Category Filter */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex gap-3 overflow-x-auto pb-2">
          <button
            onClick={() => setCategory(null)}
            className={`px-4 py-2 rounded-full whitespace-nowrap transition-all ${
              category === null
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-white text-gray-700 hover:bg-gray-100'
            }`}
          >
            All Products
          </button>
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategory(cat)}
              className={`px-4 py-2 rounded-full whitespace-nowrap transition-all ${
                category === cat
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </section>

      {/* Products Grid */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="bg-white rounded-lg shadow-md animate-pulse">
                <div className="h-64 bg-gray-200 rounded-t-lg"></div>
                <div className="p-4 space-y-3">
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                </div>
              </div>
            ))}
          </div>
        ) : products.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {products.map((product) => (
              <ProductCard key={product.id} product={product} />
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-gray-500 text-lg">No products found</p>
          </div>
        )}
      </section>

      {/* Footer */}
        <footer className="bg-gray-900 text-white py-12 mt-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div>
                <h3 className="text-lg font-bold mb-4">GeoCheapest</h3>
                <p className="text-gray-400">
                  Canada's best TCG marketplace. We find you the lowest prices from trusted vendors.
                </p>
              </div>
              <div>
                <h3 className="text-lg font-bold mb-4">For Vendors</h3>
                <ul className="space-y-2 text-gray-400">
                  <li><a href="/vendor/signup" className="hover:text-white">Sell on GeoCheapest</a></li>
                  <li><a href="/vendor/dashboard" className="hover:text-white">Vendor Dashboard</a></li>
                  <li><a href="/portal" className="hover:text-white">Portals</a></li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-bold mb-4">Support</h3>
                <ul className="space-y-2 text-gray-400">
                  <li><a href="/help" className="hover:text-white">Help Center</a></li>
                  <li><a href="/returns" className="hover:text-white">Returns</a></li>
                  <li><a href="/contact" className="hover:text-white">Contact Us</a></li>
                </ul>
              </div>
            </div>
          <div className="mt-8 pt-8 border-t border-gray-800 text-center text-gray-400">
            <p>&copy; 2024 GeoCheapest. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
