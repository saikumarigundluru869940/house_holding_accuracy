import { useEffect, useState } from 'react'
import './App.css'

const cities = ['Bengaluru', 'Chennai', 'Hyderabad', 'Kolkata', 'Mumbai', 'Pune']
const furnishing = ['Furnished', 'Semi-Furnished', 'Unfurnished']

const numberFields = [
  ['area_sqft', 'Area (sq ft)'],
  ['bedrooms', 'Bedrooms'],
  ['bathrooms', 'Bathrooms'],
  ['age_years', 'Age (years)'],
  ['floor_number', 'Floor number'],
  ['parking_spots', 'Parking spots'],
  ['distance_to_center_km', 'Distance to center (km)'],
]

export default function App() {
  const [form, setForm] = useState({ has_lift: '1' })
  const [info, setInfo] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  // Load the trained model's accuracy from the backend
  useEffect(() => {
    fetch('/api/info')
      .then((r) => r.json())
      .then(setInfo)
      .catch(() => setError('Backend not reachable. Start it with: python app.py (in the back folder).'))
  }, [])

  const change = (e) => setForm((f) => ({ ...f, [e.target.name]: e.target.value }))

  const submit = async (e) => {
    e.preventDefault()
    setResult(null)
    if (!form.city || !form.furnishing_status) return setError('Select a city and furnishing.')
    const bad = numberFields.find(([n]) => !form[n] || isNaN(Number(form[n])))
    if (bad) return setError(`Enter a valid number for "${bad[1]}".`)

    setError('')
    setLoading(true)
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const data = await res.json()
      if (!res.ok) setError(data.error || 'Prediction failed.')
      else setResult(data)
    } catch {
      setError('Backend not reachable. Start it with: python app.py (in the back folder).')
    }
    setLoading(false)
  }

  const best = info && info.metrics[info.model_used]

  return (
    <main>
      <h1>House price predictor</h1>

      {info && (
        <table className="metrics">
          <caption>Model accuracy on test data (model in use: {info.model_used})</caption>
          <thead>
            <tr><th>Model</th><th>R² score</th><th>MAE (lakhs)</th></tr>
          </thead>
          <tbody>
            {Object.entries(info.metrics).map(([name, m]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{m.r2} ({(m.r2 * 100).toFixed(1)}%)</td>
                <td>{m.mae}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <form onSubmit={submit}>
        <label>City
          <select name="city" value={form.city || ''} onChange={change}>
            <option value="">Select city</option>
            {cities.map((c) => <option key={c}>{c}</option>)}
          </select>
        </label>

        <label>Furnishing
          <select name="furnishing_status" value={form.furnishing_status || ''} onChange={change}>
            <option value="">Select furnishing</option>
            {furnishing.map((f) => <option key={f}>{f}</option>)}
          </select>
        </label>

        {numberFields.map(([name, label]) => (
          <label key={name}>{label}
            <input type="text" name={name} value={form[name] || ''} onChange={change} />
          </label>
        ))}

        <label>Lift available
          <select name="has_lift" value={form.has_lift} onChange={change}>
            <option value="1">Yes</option>
            <option value="0">No</option>
          </select>
        </label>

        <label>Actual price in lakhs (optional, to check accuracy)
          <input type="text" name="actual_price" value={form.actual_price || ''} onChange={change} />
        </label>

        <button type="submit" disabled={loading}>{loading ? 'Predicting…' : 'Predict price'}</button>
      </form>

      {error && <p className="error" role="alert">{error}</p>}

      {result && (
        <section className="result">
          <p className="price">₹{result.predicted_price_lakhs} lakhs</p>
          <p>Model accuracy (R²): {best ? (best.r2 * 100).toFixed(1) + '%' : '-'}, typical error about ₹{best ? best.mae : '-'} lakhs</p>
          {result.error_percent !== undefined && (
            <p>
              Actual ₹{result.actual_price_lakhs} lakhs. Off by ₹{Math.abs(result.error_lakhs)} lakhs
              ({result.error_percent}%), so this prediction is{' '}
              <strong>{Math.max(0, 100 - result.error_percent).toFixed(1)}% accurate</strong>.
            </p>
          )}
        </section>
      )}
    </main>
  )
}