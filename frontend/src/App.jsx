import { useState } from 'react'
import './App.css'
import Header from './components/Header';
import SentimentDisplay from './components/SentimentDisplay';
import Probabilities from './components/Probabilities'


function App() {
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  return (
    <>
      <Header/>
      <SentimentDisplay prediction={prediction} isLoading={isLoading} setPrediction={setPrediction} setIsLoading={setIsLoading}/>
      <Probabilities prediction={prediction} isLoading={isLoading}/>
    </>
  )
}

export default App
