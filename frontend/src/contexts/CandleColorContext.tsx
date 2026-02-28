import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { getConfig } from '../services/configService';

export type CandleColorMode = 'red-up' | 'green-up';

interface CandleColorContextType {
  mode: CandleColorMode;
  setMode: (mode: CandleColorMode) => void;
  upColor: string;      // hex color for up/涨
  downColor: string;    // hex color for down/跌
  upClass: string;      // tailwind class for up
  downClass: string;    // tailwind class for down
  getColorClass: (isUp: boolean) => string;
}

const COLOR_MAP: Record<CandleColorMode, { upColor: string; downColor: string; upClass: string; downClass: string }> = {
  'red-up': { upColor: '#ef4444', downColor: '#22c55e', upClass: 'text-red-500', downClass: 'text-green-500' },
  'green-up': { upColor: '#22c55e', downColor: '#ef4444', upClass: 'text-green-500', downClass: 'text-red-500' },
};

const CandleColorContext = createContext<CandleColorContextType | undefined>(undefined);

export const CandleColorProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [mode, setModeState] = useState<CandleColorMode>('red-up');

  useEffect(() => {
    getConfig().then((config) => {
      const saved = config.candleColorMode as CandleColorMode;
      if (saved && COLOR_MAP[saved]) setModeState(saved);
    }).catch(() => {});
  }, []);

  const setMode = useCallback((newMode: CandleColorMode) => {
    setModeState(newMode);
  }, []);

  const colors = COLOR_MAP[mode];

  return (
    <CandleColorContext.Provider value={{
      mode,
      setMode,
      ...colors,
      getColorClass: (isUp: boolean) => isUp ? colors.upClass : colors.downClass,
    }}>
      {children}
    </CandleColorContext.Provider>
  );
};

export const useCandleColor = () => {
  const context = useContext(CandleColorContext);
  if (!context) throw new Error('useCandleColor must be used within CandleColorProvider');
  return context;
};
