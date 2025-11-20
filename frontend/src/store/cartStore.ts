import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface CartItem {
  product_id: string
  name: string
  price: number
  quantity: number
  image_url?: string
}

interface CartStore {
  items: CartItem[]
  addItem: (item: CartItem) => void
  removeItem: (productId: string) => void
  updateQuantity: (productId: string, quantity: number) => void
  clearCart: () => void
  getTotalPrice: () => number
}

export const useCartStore = create<CartStore>()(
  persist(
    (set, get) => ({
      items: [],

      addItem: (item) =>
        set((state) => {
          const existingItem = state.items.find(
            (i) => i.product_id === item.product_id
          )

          if (existingItem) {
            return {
              items: state.items.map((i) =>
                i.product_id === item.product_id
                  ? { ...i, quantity: i.quantity + item.quantity }
                  : i
              ),
            }
          }

          return { items: [...state.items, item] }
        }),

      removeItem: (productId) =>
        set((state) => ({
          items: state.items.filter((i) => i.product_id !== productId),
        })),

      updateQuantity: (productId, quantity) =>
        set((state) => ({
          items: state.items.map((i) =>
            i.product_id === productId ? { ...i, quantity } : i
          ),
        })),

      clearCart: () => set({ items: [] }),

      getTotalPrice: () => {
        const state = get()
        return state.items.reduce(
          (sum, item) => sum + item.price * item.quantity,
          0
        )
      },
    }),
    {
      name: 'cart-storage',
    }
  )
)
