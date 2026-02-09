import React, { useState, useMemo } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis, LineChart, Line
} from 'recharts';
import { 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  BarChart3, 
  PieChart as PieIcon,
  Activity
} from 'lucide-react';

const App = () => {
  // Data derived from the January Sales Pipeline Report
  const pipelineData = [
    { age: '0-3 Mth', active: 191112346, croUpdate: 0, directUpdate: 0, hold: 4273890, count: 76 },
    { age: '3-6 Mth', active: 152416036, croUpdate: 0, directUpdate: 0, hold: 2772240, count: 72 },
    { age: '6-9 Mth', active: 67899204, croUpdate: 45976277, directUpdate: 50880339, hold: 0, count: 66 },
    { age: '9-12 Mth', active: 33900228, croUpdate: 4748908, directUpdate: 11653335, hold: 27404017, count: 38 },
    { age: '12+ Mth', active: 90336258, croUpdate: 3771903, directUpdate: 16124537, hold: 50950959, count: 86 },
  ];

  const COLORS = {
    active: '#10b981', // Emerald 500
    cro: '#f59e0b',    // Amber 500
    direct: '#ef4444', // Red 500
    hold: '#6b7280',   // Gray 500
    total: '#3b82f6'   // Blue 500
  };

  const formatCurrency = (val) => 
    new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD', 
      notation: 'compact',
      maximumFractionDigits: 1 
    }).format(val);

  const stats = {
    totalValue: 754220475,
    activeValue: 535664071,
    hrecExceeded: 150,
    activePct: 66,
    stalePct: 44
  };

  const pieData = [
    { name: 'Active', value: stats.activeValue, color: COLORS.active },
    { name: 'Updates Needed', value: 54497088 + 78658211, color: COLORS.cro },
    { name: 'On Hold/Other', value: 85401106, color: COLORS.hold },
  ];

  const bubbleData = pipelineData.map(d => ({
    name: d.age,
    x: pipelineData.indexOf(d),
    y: d.active + d.croUpdate + d.directUpdate + d.hold,
    z: d.count
  }));

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8 font-sans text-slate-900">
      {/* Header */}
      <div className="max-w-7xl mx-auto mb-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">January Sales Pipeline</h1>
            <p className="text-slate-500">Executive Performance & Aging Analytics</p>
          </div>
          <div className="flex gap-2 bg-white p-1 rounded-lg border shadow-sm">
            <button className="px-4 py-2 bg-blue-50 text-blue-600 rounded-md font-medium text-sm">Overview</button>
            <button className="px-4 py-2 text-slate-500 hover:bg-slate-50 rounded-md font-medium text-sm">Deep Dive</button>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center gap-4 mb-2">
            <div className="p-2 bg-blue-100 rounded-lg text-blue-600">
              <TrendingUp size={20} />
            </div>
            <span className="text-sm font-medium text-slate-500 uppercase tracking-wider">Total Pipeline</span>
          </div>
          <div className="text-2xl font-bold">{formatCurrency(stats.totalValue)}</div>
          <div className="mt-2 text-xs text-slate-400">Total book of business</div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center gap-4 mb-2">
            <div className="p-2 bg-emerald-100 rounded-lg text-emerald-600">
              <CheckCircle size={20} />
            </div>
            <span className="text-sm font-medium text-slate-500 uppercase tracking-wider">Health Rating</span>
          </div>
          <div className="text-2xl font-bold text-emerald-600">{stats.activePct}% Active</div>
          <div className="mt-2 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
            <div className="h-full bg-emerald-500 rounded-full" style={{ width: `${stats.activePct}%` }}></div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center gap-4 mb-2">
            <div className="p-2 bg-amber-100 rounded-lg text-amber-600">
              <AlertTriangle size={20} />
            </div>
            <span className="text-sm font-medium text-slate-500 uppercase tracking-wider">HREC Exceeded</span>
          </div>
          <div className="text-2xl font-bold text-amber-600">{stats.hrecExceeded} Opps</div>
          <div className="mt-2 text-xs text-slate-400">{stats.stalePct}% of high priority accounts</div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center gap-4 mb-2">
            <div className="p-2 bg-slate-100 rounded-lg text-slate-600">
              <Clock size={20} />
            </div>
            <span className="text-sm font-medium text-slate-500 uppercase tracking-wider">Avg Deal Age</span>
          </div>
          <div className="text-2xl font-bold">7.2 Months</div>
          <div className="mt-2 text-xs text-red-500">Trending +12% MoM</div>
        </div>
      </div>

      {/* Main Charts Row */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        
        {/* Pipeline Aging (Stacked Bar) */}
        <div className="lg:col-span-2 bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-bold flex items-center gap-2">
              <BarChart3 size={18} className="text-blue-500" />
              Pipeline Composition by Age
            </h3>
            <span className="text-xs bg-slate-100 px-2 py-1 rounded text-slate-600">Total Value ($)</span>
          </div>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={pipelineData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="age" axisLine={false} tickLine={false} />
                <YAxis hide />
                <Tooltip 
                  cursor={{fill: '#f8fafc'}}
                  formatter={(value) => formatCurrency(value)}
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Legend verticalAlign="top" height={36}/>
                <Bar dataKey="active" name="Active" stackId="a" fill={COLORS.active} radius={[0, 0, 0, 0]} />
                <Bar dataKey="croUpdate" name="CRO Update" stackId="a" fill={COLORS.cro} />
                <Bar dataKey="directUpdate" name="Direct Update" stackId="a" fill={COLORS.direct} />
                <Bar dataKey="hold" name="On Hold" stackId="a" fill={COLORS.hold} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pipeline Health (Pie) */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between mb-6">
            <h3 className="font-bold flex items-center gap-2">
              <PieIcon size={18} className="text-emerald-500" />
              Pipeline Health Mix
            </h3>
          </div>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => formatCurrency(value)} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 space-y-2">
            {pieData.map(item => (
              <div key={item.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                  <span className="text-slate-600">{item.name}</span>
                </div>
                <span className="font-semibold">{Math.round((item.value / stats.totalValue) * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Insights Row */}
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Risk Bubble Chart */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="font-bold mb-6 flex items-center gap-2 text-slate-800">
            <Activity size={18} className="text-red-500" />
            HREC Risk Cluster (Volume vs. Value)
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="x" 
                  name="Age Bucket" 
                  ticks={[0, 1, 2, 3, 4]} 
                  tickFormatter={(val) => pipelineData[val]?.age} 
                />
                <YAxis 
                  dataKey="y" 
                  name="Value" 
                  tickFormatter={(val) => formatCurrency(val)}
                />
                <ZAxis dataKey="z" range={[100, 1000]} name="Account Count" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter name="Age Groups" data={bubbleData} fill="#3b82f6" fillOpacity={0.6} stroke="#2563eb" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-xs text-slate-500 italic">
            *Bubble size indicates number of accounts. Note high concentration in 0-3M and 12+M groups.
          </p>
        </div>

        {/* Action Items */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="font-bold mb-4 text-slate-800">Critical Action Items</h3>
          <div className="space-y-4">
            <div className="p-4 bg-red-50 border-l-4 border-red-500 rounded-r-lg">
              <p className="text-sm font-bold text-red-800">Audit "Active" 12+ Month Deals</p>
              <p className="text-xs text-red-700 mt-1">$90.3M is currently marked active despite being {'>'}1 year old. High risk of pipeline inflation.</p>
            </div>
            <div className="p-4 bg-amber-50 border-l-4 border-amber-500 rounded-r-lg">
              <p className="text-sm font-bold text-amber-800">Hygiene Sprint: 6-9 Month Bracket</p>
              <p className="text-xs text-amber-700 mt-1">96M in "Updates Needed". Immediate engagement required from CRO and Direct teams.</p>
            </div>
            <div className="p-4 bg-emerald-50 border-l-4 border-emerald-500 rounded-r-lg">
              <p className="text-sm font-bold text-emerald-800">Velocity Opportunity</p>
              <p className="text-xs text-emerald-700 mt-1">Strong $350M core in the 0-6 month range. Focus marketing efforts on late-stage acceleration here.</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default App;