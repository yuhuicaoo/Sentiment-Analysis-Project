import { useState } from 'react'
import './App.css'
import Header from './components/Header';
import SentimentDisplay from './components/SentimentDisplay';


function App() {
  return (
    <>
      <Header/>
      <SentimentDisplay/>
    </>
  )
}

export default App
