<div><h3>Analysis Information</h3><div><h3>Multiple Experiment Information</h3><div><h3>Experiment Information</h3><p><strong>Test Key:</strong> None</p><p><strong>Experiment ID:</strong> None</p><p><strong>Segment ID:</strong> None</p><p><strong>Hash ID:</strong> None</p><p><strong>Start Date:</strong> None</p><p><strong>End Date:</strong> None</p><p><strong>Variants Mapping:</strong> None</p><p><strong>Variants:</strong> None</p><p><strong>Control Label:</strong> None</p><p><strong>Treatment Label:</strong> enabled</p><p><strong>Query:</strong> None</p><h4>Derived Stats:</h4><p>None</p></div><div><h3>Experiment Information</h3><p><strong>Test Key:</strong> None</p><p><strong>Experiment ID:</strong> None</p><p><strong>Segment ID:</strong> None</p><p><strong>Hash ID:</strong> None</p><p><strong>Start Date:</strong> None</p><p><strong>End Date:</strong> None</p><p><strong>Variants Mapping:</strong> None</p><p><strong>Variants:</strong> None</p><p><strong>Control Label:</strong> None</p><p><strong>Treatment Label:</strong> enabled</p><p><strong>Query:</strong> None</p><h4>Derived Stats:</h4><p>None</p></div><p><strong>Merge Method:</strong> cross</p><h4>Derived Stats:</h4><div><h3>Derived Experiment Statistics</h3><p><strong>Variants:</strong> Variant(value=('control', 'control'), name='(control, control)'), Variant(value=('control', 'nan'), name='(control, nan)'), Variant(value=('control', 'test'), name='(control, test)'), Variant(value=('nan', 'control'), name='(nan, control)'), Variant(value=('nan', 'test'), name='(nan, test)'), Variant(value=('v1', 'control'), name='(v1, control)'), Variant(value=('v1', 'nan'), name='(v1, nan)'), Variant(value=('v1', 'test'), name='(v1, test)'), Variant(value=('v2', 'control'), name='(v2, control)'), Variant(value=('v2', 'nan'), name='(v2, nan)'), Variant(value=('v2', 'test'), name='(v2, test)')</p><p><strong>Launches:</strong> Launch(value=('control', 'control'), name='(control, control)'), Launch(value=('control', 'test'), name='(control, test)'), Launch(value=('v1', 'control'), name='(v1, control)'), Launch(value=('v1', 'test'), name='(v1, test)'), Launch(value=('v2', 'control'), name='(v2, control)'), Launch(value=('v2', 'test'), name='(v2, test)')</p><p><strong>Non-Control Launches:</strong> Launch(value=('control', 'test'), name='(control, test)'), Launch(value=('v1', 'control'), name='(v1, control)'), Launch(value=('v1', 'test'), name='(v1, test)'), Launch(value=('v2', 'control'), name='(v2, control)'), Launch(value=('v2', 'test'), name='(v2, test)')</p><p><strong>Trigger States:</strong> TriggerState(value=(True, True), overall_value=True, name='(True, True)'), TriggerState(value=(True, False), overall_value=True, name='(True, False)'), TriggerState(value=(False, True), overall_value=True, name='(False, True)')</p><h4>Variant Count DataFrame:</h4><table border="1" class="dataframe">
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
      <td>877</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>5405</td>
      <td>8.490657</td>
      <td>52.328396</td>
      <td>16.225717</td>
    </tr>
    <tr>
      <th>(control, nan)</th>
      <td>816</td>
      <td>(True, False)</td>
      <td>True</td>
      <td>2452</td>
      <td>7.900087</td>
      <td>23.738987</td>
      <td>33.278956</td>
    </tr>
    <tr>
      <th>(control, test)</th>
      <td>919</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>5405</td>
      <td>8.897280</td>
      <td>52.328396</td>
      <td>17.002775</td>
    </tr>
    <tr>
      <th>(nan, control)</th>
      <td>1263</td>
      <td>(False, True)</td>
      <td>True</td>
      <td>2472</td>
      <td>12.227708</td>
      <td>23.932617</td>
      <td>51.092233</td>
    </tr>
    <tr>
      <th>(nan, test)</th>
      <td>1209</td>
      <td>(False, True)</td>
      <td>True</td>
      <td>2472</td>
      <td>11.704909</td>
      <td>23.932617</td>
      <td>48.907767</td>
    </tr>
    <tr>
      <th>(v1, control)</th>
      <td>823</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>5405</td>
      <td>7.967857</td>
      <td>52.328396</td>
      <td>15.226642</td>
    </tr>
    <tr>
      <th>(v1, nan)</th>
      <td>859</td>
      <td>(True, False)</td>
      <td>True</td>
      <td>2452</td>
      <td>8.316391</td>
      <td>23.738987</td>
      <td>35.032626</td>
    </tr>
    <tr>
      <th>(v1, test)</th>
      <td>873</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>5405</td>
      <td>8.451931</td>
      <td>52.328396</td>
      <td>16.151711</td>
    </tr>
    <tr>
      <th>(v2, control)</th>
      <td>921</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>5405</td>
      <td>8.916642</td>
      <td>52.328396</td>
      <td>17.039778</td>
    </tr>
    <tr>
      <th>(v2, nan)</th>
      <td>777</td>
      <td>(True, False)</td>
      <td>True</td>
      <td>2452</td>
      <td>7.522509</td>
      <td>23.738987</td>
      <td>31.688418</td>
    </tr>
    <tr>
      <th>(v2, test)</th>
      <td>992</td>
      <td>(True, True)</td>
      <td>True</td>
      <td>5405</td>
      <td>9.604027</td>
      <td>52.328396</td>
      <td>18.353377</td>
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
      <td>2472</td>
      <td>23.932617</td>
    </tr>
    <tr>
      <th>(True, False)</th>
      <td>2452</td>
      <td>23.738987</td>
    </tr>
    <tr>
      <th>(True, True)</th>
      <td>5405</td>
      <td>52.328396</td>
    </tr>
  </tbody>
</table><p><strong>Total Count:</strong> 10329</p><p><strong>Total Triggered Count:</strong> 10329</p><p><strong>Total Triggered Percent:</strong> 100.0</p><p><strong>Conditional Trigger Dfs:</strong></p><p><p>expt 1's overlap by other:</p> <p><table border="1" class="dataframe">
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
      <td>2452</td>
      <td>31.20784</td>
    </tr>
    <tr>
      <th>(True, True)</th>
      <td>5405</td>
      <td>68.79216</td>
    </tr>
  </tbody>
</table></p><p>expt 2's overlap by other:</p> <p><table border="1" class="dataframe">
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
      <td>2472</td>
      <td>31.382506</td>
    </tr>
    <tr>
      <th>(True, True)</th>
      <td>5405</td>
      <td>68.617494</td>
    </tr>
  </tbody>
</table></p></p><p><strong>Overlap Rates:</strong><p><p><table>
  <thead>
    <tr><th>Expt</th><th>Overlap Rate (%)</th></tr>
  </thead>
  <tbody>
    <tr><td>Expt 1's overlap: </td><td>68.79%</td></tr>
    <tr><td>Expt 2's overlap: </td><td>68.62%</td></tr>
  </tbody>
</table></p></div></div><p><strong>Metric Family:</strong> </p><p><strong>Analysis Start Date:</strong> None</p><p><strong>Analysis End Date:</strong> None</p></div>