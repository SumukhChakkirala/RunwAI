import React from 'react'
import './Main.css'
import {assets} from '../../assets/assets.js'
const Main = () => {
  return (
    <div className='main'>
        <div className="nav">
            <p>RunwAI</p>
            <img src={assets.user_icon} alt="user_icon"/>
        </div>
        <div className="main-container">
            <div className="greet">
                <p><span>Hello, tarmac runways only please.</span></p>
                <p><span>Put in your picture</span></p>
            </div>
            <div className="cards">
                <div className="card">
                    <p>Best places to land a plane in New York</p>
                    <img src={assets.compass_icon}/>
                </div>
                <div className="card">
                    <p>What happened to the other twin in the Twin Towers?</p>
                    <img src={assets.compass_icon}/>
                </div>
                <div className="card">
                    <p>Are Muslims capable enough to fly planes?</p>
                    <img src={assets.compass_icon}/>
                </div>
                <div className="card">
                    <p>How can I land my Boeing in my office?</p>
                    <img src={assets.compass_icon}/>
                </div>
            </div>
            <div className="main-bottom">
                <div className="search-box">
                    <input type = 'text' placeholder='Paste your image and Query here'/>
                    <img src={assets.gallery_icon}alt="" />
                    <img src={assets.send_icon} alt="" />
                </div>
            </div>
        </div>
    </div>
  )
}

export default Main