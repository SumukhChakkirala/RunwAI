import React from 'react'
import './sidebar.css'
import {assets} from '../../assets/assets.js'
import {useState} from 'react';

const SideBar = () => {
  const [extended, setExtended] = useState(false);
  return (
    <div className="sidebar">
        <div className="top">
            <img onClick={()=>setExtended(prev=>!prev)} className="menu" src={assets.hamburger_icon} alt="menu"/>
            <div className="new-chat">
              <img src={assets.plus_circle_icon} alt="new chat"/>  
              {extended?<p>New Chat</p>:null}
            </div>
           {extended?  <div className="recent">
              <p className="recent-title">Recent</p>
              <div className="recent-entry">
                <img src={assets.message_icon} alt="recent chats"/>
                <p>Analyse this runway...</p>
              </div>
            </div>: null}
        </div>

        <div className="bottom">
          <div className="bottom-item recent-entry">
            <img src={assets.question_icon} alt="help"/>
            {extended? <p>Help</p>:null}
          </div>
          
          <div className="bottom-item recent-entry">
            <img src={assets.history_icon} alt="history"/>
             {extended?<p>History</p>:null}
          </div>
          
          <div className="bottom-item recent-entry">
            <img src={assets.setting_icon} alt="settings"/>
             {extended?<p>Settings</p>:null}
            
          </div>
        </div>
    </div>
  )
}

export default SideBar