<div><h3>Analysis Information</h3><div><h3>Multiple Experiment Information</h3><div><h3>Experiment Information</h3><p><strong>Test Key:</strong> None</p><p><strong>Experiment ID:</strong> None</p><p><strong>Segment ID:</strong> None</p><p><strong>Hash ID:</strong> None</p><p><strong>Start Date:</strong> None</p><p><strong>End Date:</strong> None</p><p><strong>Variants Mapping:</strong> None</p><p><strong>Variants:</strong> None</p><p><strong>Control Label:</strong> None</p><p><strong>Treatment Label:</strong> enabled</p><p><strong>Query:</strong> None</p><h4>Derived Stats:</h4><p>None</p></div><div><h3>Experiment Information</h3><p><strong>Test Key:</strong> None</p><p><strong>Experiment ID:</strong> None</p><p><strong>Segment ID:</strong> None</p><p><strong>Hash ID:</strong> None</p><p><strong>Start Date:</strong> None</p><p><strong>End Date:</strong> None</p><p><strong>Variants Mapping:</strong> None</p><p><strong>Variants:</strong> None</p><p><strong>Control Label:</strong> None</p><p><strong>Treatment Label:</strong> enabled</p><p><strong>Query:</strong> None</p><h4>Derived Stats:</h4><p>None</p></div><p><strong>Merge Method:</strong> cross</p><h4>Derived Stats:</h4><div><h3>Derived Experiment Statistics</h3><p><strong>Variants:</strong> Variant(value=('control', 'control'), name='(control, control)'), Variant(value=('control', 'enabled'), name='(control, enabled)'), Variant(value=('control', 'nan'), name='(control, nan)'), Variant(value=('nan', 'control'), name='(nan, control)'), Variant(value=('nan', 'enabled'), name='(nan, enabled)'), Variant(value=('v1', 'control'), name='(v1, control)'), Variant(value=('v1', 'enabled'), name='(v1, enabled)'), Variant(value=('v1', 'nan'), name='(v1, nan)'), Variant(value=('v2', 'control'), name='(v2, control)'), Variant(value=('v2', 'enabled'), name='(v2, enabled)'), Variant(value=('v2', 'nan'), name='(v2, nan)')</p><p><strong>Launches:</strong> Launch(value=('control', 'control'), name='(control, control)'), Launch(value=('control', 'enabled'), name='(control, enabled)'), Launch(value=('v1', 'control'), name='(v1, control)'), Launch(value=('v1', 'enabled'), name='(v1, enabled)'), Launch(value=('v2', 'control'), name='(v2, control)'), Launch(value=('v2', 'enabled'), name='(v2, enabled)')</p><p><strong>Non-Control Launches:</strong> Launch(value=('control', 'enabled'), name='(control, enabled)'), Launch(value=('v1', 'control'), name='(v1, control)'), Launch(value=('v1', 'enabled'), name='(v1, enabled)'), Launch(value=('v2', 'control'), name='(v2, control)'), Launch(value=('v2', 'enabled'), name='(v2, enabled)')</p><p><strong>Trigger States:</strong> TriggerState(value=(True, True), overall_value=True, name='(True, True)'), TriggerState(value=(True, False), overall_value=True, name='(True, False)'), TriggerState(value=(False, True), overall_value=True, name='(False, True)')</p><h4>Variant Count DataFrame:</h4><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_count</th>
      <th>trigger_state</th>
      <th>trigger_state_overall</th>
      <th>trigger_state_count</th>
      <th>variant_percent</th>
      <th>trigger_state_percent</th>
      <th>variant_over_triggered_pcnt</th>
    </tr>
    <tr>
      <th>variant</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(control, control)</th>
      <td>102</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>920</td>
      <td>10.2</td>
      <td>92.0</td>
      <td>11.086957</td>
    </tr>
    <tr>
      <th>(control, enabled)</th>
      <td>253</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>920</td>
      <td>25.3</td>
      <td>92.0</td>
      <td>27.500000</td>
    </tr>
    <tr>
      <th>(control, nan)</th>
      <td>20</td>
      <td>(True, False)</td>
      <td>True</td>
      <td>46</td>
      <td>2.0</td>
      <td>4.6</td>
      <td>43.478261</td>
    </tr>
    <tr>
      <th>(nan, control)</th>
      <td>11</td>
      <td>(False, True)</td>
      <td>True</td>
      <td>34</td>
      <td>1.1</td>
      <td>3.4</td>
      <td>32.352941</td>
    </tr>
    <tr>
      <th>(nan, enabled)</th>
      <td>23</td>
      <td>(False, True)</td>
      <td>True</td>
      <td>34</td>
      <td>2.3</td>
      <td>3.4</td>
      <td>67.647059</td>
    </tr>
    <tr>
      <th>(v1, control)</th>
      <td>88</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>920</td>
      <td>8.8</td>
      <td>92.0</td>
      <td>9.565217</td>
    </tr>
    <tr>
      <th>(v1, enabled)</th>
      <td>209</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>920</td>
      <td>20.9</td>
      <td>92.0</td>
      <td>22.717391</td>
    </tr>
    <tr>
      <th>(v1, nan)</th>
      <td>14</td>
      <td>(True, False)</td>
      <td>True</td>
      <td>46</td>
      <td>1.4</td>
      <td>4.6</td>
      <td>30.434783</td>
    </tr>
    <tr>
      <th>(v2, control)</th>
      <td>81</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>920</td>
      <td>8.1</td>
      <td>92.0</td>
      <td>8.804348</td>
    </tr>
    <tr>
      <th>(v2, enabled)</th>
      <td>187</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>920</td>
      <td>18.7</td>
      <td>92.0</td>
      <td>20.326087</td>
    </tr>
    <tr>
      <th>(v2, nan)</th>
      <td>12</td>
      <td>(True, False)</td>
      <td>True</td>
      <td>46</td>
      <td>1.2</td>
      <td>4.6</td>
      <td>26.086957</td>
    </tr>
  </tbody>
</table><h4>Trigger State Count DataFrame:</h4><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trigger_state_count</th>
      <th>trigger_state_percent</th>
    </tr>
    <tr>
      <th>trigger_state</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(False, True)</th>
      <td>34</td>
      <td>3.4</td>
    </tr>
    <tr>
      <th>(True, False)</th>
      <td>46</td>
      <td>4.6</td>
    </tr>
    <tr>
      <th>(True, True)</th>
      <td>920</td>
      <td>92.0</td>
    </tr>
  </tbody>
</table><p><strong>Total Count:</strong> 1000</p><p><strong>Total Triggered Count:</strong> 1000</p><p><strong>Total Triggered Percent:</strong> 100.0</p><p><strong>Conditional Trigger Dfs:</strong></p><p><p>expt 1's interference by other:</p> <p><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trigger_state_count</th>
      <th>trigger_state_percent</th>
    </tr>
    <tr>
      <th>trigger_state</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(True, False)</th>
      <td>46</td>
      <td>4.761905</td>
    </tr>
    <tr>
      <th>(True, True)</th>
      <td>920</td>
      <td>95.238095</td>
    </tr>
  </tbody>
</table></p><p>expt 2's interference by other:</p> <p><table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trigger_state_count</th>
      <th>trigger_state_percent</th>
    </tr>
    <tr>
      <th>trigger_state</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(False, True)</th>
      <td>34</td>
      <td>3.563941</td>
    </tr>
    <tr>
      <th>(True, True)</th>
      <td>920</td>
      <td>96.436059</td>
    </tr>
  </tbody>
</table></p></p><p><strong>Interference Rates:</strong><p><p><table>
  <thead>
    <tr><th>Expt</th><th>Interference Rate (%)</th></tr>
  </thead>
  <tbody>
    <tr><td>Expt 1's interference: </td><td>95.24%</td></tr>
    <tr><td>Expt 2's interference: </td><td>96.44%</td></tr>
  </tbody>
</table></p></div></div><p><strong>Metric Family:</strong> </p><p><strong>Analysis Start Date:</strong> None</p><p><strong>Analysis End Date:</strong> None</p></div>