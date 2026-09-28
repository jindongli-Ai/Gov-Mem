from scripts.summarize_v8_exact_bootstrap import exact_interval


def test_exact_bootstrap_enumerates_ordered_episode_samples():
    def episode(ep,correct):
        return [{'episode_id':ep,'checkpoint_id':ep+q,'query_type':q,
                 'judge':{'parse_ok':True,'action_ok':True,key:val}}
                for q,key,val in [('utility','utility_ok',correct),('privacy','privacy_leak',False),('safety','deletion_leak',False)]]
    full={'d':episode('a',True)+episode('b',True)}
    control={'d':episode('a',False)+episode('b',True)}
    manifest={'domains':{'d':{'selected_episodes':{'a':3,'b':3}}}}
    result=exact_interval(full,control,manifest)
    assert result=={'interval':[0.0,1.0],'ordered_resamples':4}
    assert exact_interval(full,full,manifest)['interval']==[0,0]
