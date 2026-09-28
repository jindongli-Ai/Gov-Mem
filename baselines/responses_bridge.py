"""Protocol-only adapter for OpenLux nano; preserves text and generation settings."""
def to_responses(payload):
    allowed = {'model', 'messages', 'temperature', 'max_tokens', 'max_completion_tokens',
               'response_format', 'top_p', 'store', 'stream'}
    if set(payload) - allowed:
        raise ValueError('Unsupported nano chat fields: ' + str(sorted(set(payload) - allowed)))
    if payload.get('stream'):
        raise ValueError('Streaming not supported by experiment adapter')
    messages = payload['messages']
    if any(set(m) - {'role', 'content'} or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Only unchanged text messages supported')
    out = {'model': payload['model'], 'input': messages,
           'max_output_tokens': payload.get('max_completion_tokens', payload.get('max_tokens', 4096))}
    for key in ('temperature', 'top_p', 'store'):
        if key in payload:
            out[key] = payload[key]
    if payload.get('response_format'):
        fmt = payload['response_format']
        if fmt['type'] not in ('json_object', 'text'):
            raise ValueError('Unsupported response format')
        out['text'] = {'format': fmt}
    return out


def to_chat(body):
    if body.get('status') != 'completed':
        raise ValueError('Responses did not complete: ' + str(body.get('status')))
    chunks = []
    for item in body.get('output', []):
        if item.get('type') == 'message':
            for part in item.get('content', []):
                if part.get('type') == 'output_text':
                    chunks.append(part['text'])
    if not chunks:
        raise ValueError('Responses returned no output text')
    usage = body.get('usage')
    mapped = None if usage is None else {
        'prompt_tokens': usage.get('input_tokens', 0),
        'completion_tokens': usage.get('output_tokens', 0),
        'total_tokens': usage.get('total_tokens', 0),
        'prompt_tokens_details': usage.get('input_tokens_details', {}),
        'completion_tokens_details': usage.get('output_tokens_details', {}),
    }
    return {'id': body.get('id'), 'object': 'chat.completion', 'model': body.get('model'),
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''.join(chunks)},
                         'finish_reason': 'stop'}], 'usage': mapped}
