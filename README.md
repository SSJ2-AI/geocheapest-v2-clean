# GeoCheapest v2 - TCG Marketplace Platform

Complete marketplace platform for trading card game (TCG) products in Canada. Compares prices from multiple vendors (Shopify stores + affiliates) to help customers find the best deals.

## 🏗️ Architecture (RoadSense Method)

- **Backend**: FastAPI on Google Cloud Run
- **Frontend**: Next.js (static export) on Firebase Hosting
- **Database**: Google Cloud Firestore
- **CI/CD**: GitHub Actions with Workload Identity Federation
- **Payments**: Stripe + Stripe Connect
- **Shipping**: Shippo API

## 🎯 Features

### Customer Features
- ✅ Unified product display showing best prices from all sources
- ✅ Cart optimization with shipping calculation
- ✅ Split checkout (Stripe for Shopify, redirect for affiliates)
- ✅ Real-time price comparison
- ✅ Pre-order support
- ✅ Returns & cancellations
- ✅ Registered user portal for saved cards & order history

### Vendor Features
- ✅ Shopify OAuth integration
- ✅ Automatic product sync
- ✅ Sales dashboard & analytics
- ✅ Payout tracking (Stripe Connect)
- ✅ Shippo label generation
- ✅ Return request management

### Admin Features
- ✅ Platform analytics dashboard
- ✅ Vendor approval system
- ✅ Configurable commission rates (per category & per vendor)
- ✅ Payment gateway fee configuration
- ✅ Return/refund management

## 🚀 Deployment

### Prerequisites

1. **GCP Project Setup**
   ```bash
   gcloud config set project geocheapest
   gcloud services enable run.googleapis.com
   gcloud services enable firestore.googleapis.com
   gcloud services enable secretmanager.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   ```

2. **Create Artifact Registry Repository**
   ```bash
   gcloud artifacts repositories create geocheapest \
     --repository-format=docker \
     --location=us-central1 \
     --description="GeoCheapest container images"
   ```

3. **Configure Secrets**
   ```bash
    # Shopify
    echo -n "your_shopify_api_key" | gcloud secrets create shopify-api-key --data-file=-
    echo -n "your_shopify_api_secret" | gcloud secrets create shopify-api-secret --data-file=-
    
    # Stripe
    echo -n "sk_live_xxx" | gcloud secrets create stripe-secret-key --data-file=-
    echo -n "whsec_xxx" | gcloud secrets create stripe-webhook-secret --data-file=-
    
    # Shippo
    echo -n "your_shippo_api_token" | gcloud secrets create shippo-api-key --data-file=-
    
    # Affiliate APIs
    echo -n "your_rapidapi_key" | gcloud secrets create rapidapi-key --data-file=-
    
    # Admin
    echo -n "super_secret_admin_key" | gcloud secrets create admin-api-key --data-file=-
   ```

4. **Setup Workload Identity Federation**
   ```bash
   # Create service account
   gcloud iam service-accounts create github-actions \
     --display-name="GitHub Actions"
   
   # Grant necessary permissions
   gcloud projects add-iam-policy-binding geocheapest \
     --member="serviceAccount:github-actions@geocheapest.iam.gserviceaccount.com" \
     --role="roles/run.admin"
   
   gcloud projects add-iam-policy-binding geocheapest \
     --member="serviceAccount:github-actions@geocheapest.iam.gserviceaccount.com" \
     --role="roles/iam.serviceAccountUser"
   
   # Create workload identity pool
   gcloud iam workload-identity-pools create github \
     --location="global" \
     --display-name="GitHub Actions Pool"
   
   # Create provider
   gcloud iam workload-identity-pools providers create-oidc github \
     --location="global" \
     --workload-identity-pool="github" \
     --display-name="GitHub Provider" \
     --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
     --issuer-uri="https://token.actions.githubusercontent.com"
   ```

5. **GitHub Repository Secrets**
   Add these secrets to your GitHub repository:
   - `GCP_PROJECT_ID`: Your GCP project ID
   - `GCP_WORKLOAD_IDENTITY_PROVIDER`: Full workload identity provider name
   - `GCP_SERVICE_ACCOUNT`: github-actions@geocheapest.iam.gserviceaccount.com
   - `FIREBASE_PROJECT_ID`: Your Firebase project ID
   - `FIREBASE_SERVICE_ACCOUNT`: Firebase service account JSON
   - `STRIPE_PUBLISHABLE_KEY`: pk_live_xxx

### Automatic Deployment

Push to `main` branch or create a PR to trigger automatic deployment:

```bash
git push origin main
```

The GitHub Actions workflow will:
1. Build and deploy backend to Cloud Run
2. Build and deploy frontend to Firebase Hosting
3. Seed initial products (first deploy only)

### Manual Deployment

#### Backend
```bash
cd backend
gcloud builds submit --config=../cloudbuild.yaml
```

#### Frontend
```bash
cd frontend
npm ci
npm run build
firebase deploy --only hosting
```

### Seed Initial Products
```bash
cd backend
pip install -r requirements.txt
python seed_data.py
```

## 🔧 Local Development

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Set environment variables
export GCP_PROJECT_ID=geocheapest
export SHOPIFY_API_KEY=your_key
export SHOPIFY_API_SECRET=your_secret
# ... other vars from .env.example

uvicorn main:app --reload --port 8000
```

### Stripe Configuration & Environment

Add these variables to your `.env` (or platform secrets) to enable real Stripe Connect flows:

| Variable | Description |
| --- | --- |
| `STRIPE_SECRET_KEY` | Live mode secret key (sk_live_...) used by the backend |
| `STRIPE_PUBLISHABLE_KEY` | Live mode publishable key for the frontend |
| `STRIPE_WEBHOOK_SECRET` | Signing secret from the Stripe webhook endpoint |
| `FRONTEND_URL` / `BACKEND_URL` | Fully-qualified URLs used for Checkout success/cancel + OAuth redirects |
| `SHOPIFY_TOKEN_ENCRYPTION_KEY` | 32-byte Fernet key (base64) for encrypting vendor OAuth tokens |
| `STRIPE_PLATFORM_COMMISSION_SEALED` | Decimal platform commission for sealed products (default `0.045`) |
| `STRIPE_PLATFORM_COMMISSION_SINGLES` | Decimal platform commission for singles (`0.02`) |
| `STRIPE_CARD_PERCENT_FEE` | Percent processing fee portion (default `0.029`) |
| `STRIPE_CARD_FIXED_FEE` | Fixed processing fee in CAD dollars (default `0.30`) |
| `STRIPE_VENDOR_SUB_BASIC_PRICE_ID` | Stripe Price ID for the $29 vendor tier |
| `STRIPE_VENDOR_SUB_GROWTH_PRICE_ID` | Stripe Price ID for the $79 vendor tier |
| `STRIPE_VENDOR_SUB_PRO_PRICE_ID` | Stripe Price ID for the $199 vendor tier |

**Webhook endpoints**

Create a live-mode webhook at `https://geocheapest-api.onrender.com/api/stripe/webhook` and subscribe to:

- `checkout.session.completed`
- `payment_intent.payment_failed`
- `invoice.payment_succeeded`
- `invoice.payment_failed`
- `customer.subscription.deleted`

Be sure to mirror the same configuration in your local tunnel when running the backend locally (e.g., with `stripe listen --forward-to localhost:8000/api/stripe/webhook`).

### Amazon Affiliate Configuration

| Variable | Description |
| --- | --- |
| `AMAZON_CA_AFFILIATE_TAG` | Affiliate tag appended to every Amazon.ca link |
| `AMAZON_SYNC_INTERVAL_SECONDS` | Cadence for the background Amazon sync loop (default `86400`) |
| `AMAZON_MIN_RATING` | Minimum product rating to ingest (default `4.0`) |
| `AMAZON_MIN_REVIEWS` | Minimum verified reviews (default `50`) |

The backend runs a background cron loop (triggered automatically at startup) that queries the RapidAPI Amazon Data API for every supported TCG franchise, filters for ≥4★ / 50+ reviews, and stores sealed listings in `affiliateProducts` with real ratings, inventory estimates, and `geocheapest-20` affiliate links.

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

## 📊 Database Schema (Firestore)

### Collections
- `stores` - Shopify vendor stores
- `products` - Canonical products (unified listings)
- `shopifyListings` - Product listings from Shopify stores
- `affiliateProducts` - Products from Amazon.ca (and future affiliate sources)
- `orders` - Customer orders
- `orderItems` - Individual order line items
- `sellerPayouts` - Vendor payout records
- `commissionRates` - Platform commission configuration
- `vendorCommissionOverrides` - Custom rates per vendor
- `returnRequests` - Return/refund requests

## 💳 Business Model

### Revenue Stream 1: Shopify Vendors
- Vendors connect via Shopify OAuth
- Customers checkout on GeoCheapest (Stripe)
- Platform keeps: Stripe fee (2.9% + $0.30) + Commission (8-10%)
- Vendor receives net payout via Stripe Connect

### Revenue Stream 2: Affiliates
- Amazon.ca (geocheapest-20 tag, 3% commission)
- Amazon Associates (5% commission)
- Customers redirect to affiliate sites
- Platform earns commission on sales

## 🛠️ Tech Stack

### Backend
- FastAPI 0.109
- Google Cloud Firestore
- Stripe 7.10
- httpx (async HTTP)
- Pydantic v2

### Frontend
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Zustand (state management)
- Axios
- Lucide React (icons)

### Infrastructure
- Google Cloud Run (backend)
- Firebase Hosting (frontend)
- Google Cloud Firestore (database)
- Google Secret Manager (secrets)
- GitHub Actions (CI/CD)

## 🎨 Starter Products

7 products seeded on first deployment:

**Direct (Amazon affiliate)**
1. Pokemon 151 Booster Box ($149.91)
2. One Piece Two Legends Booster Box ($199.99)
3. Pokemon White Flare ETB ($146.87)

**Affiliate (Amazon.ca)**
4. One Piece Legacy Premium Collection ($199.99)
5. Pokemon Black Bolt ETB ($120.02)
6. One Piece Royal Blood Starter Deck ($88.00)
7. Pokemon Destined Rivals Premium Collection ($649.99)

## 📝 API Documentation

Once deployed, visit:
- API Docs: https://your-api-url.run.app/docs
- ReDoc: https://your-api-url.run.app/redoc

## 🔐 Security

- Shopify OAuth webhooks verified with HMAC
- Stripe webhooks verified with signatures
- Admin portal protected by API key
- Secrets stored in Google Secret Manager
- CORS configured for production domains

## 📈 Monitoring

- Cloud Run metrics in GCP Console
- Firebase Hosting analytics
- Stripe Dashboard for payments
- Shippo for shipping tracking

## 🤝 Contributing

This platform uses the RoadSense AI methodology:
1. Cloud-native from day one
2. NO Vercel
3. FastAPI + Firestore + Cloud Run + Firebase
4. GitHub Actions for CI/CD

## 📄 License

See LICENSE file.

## 🆘 Support

For vendor onboarding or technical support, contact support@geocheapest.com

---

Built with ❤️ for the Canadian TCG community
