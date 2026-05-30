<?php

namespace Pankaj\MHFSafeguard\Pipeline;

use Pankaj\MHFSafeguard\Content\ContentContext;

class PayloadBuilder
{
    public function build(ContentContext $context, string $cleanMessage): array
    {
        $options = \XF::options();
        $boardUrl = isset($options->boardUrl) ? (string)$options->boardUrl : '';

        return [
            'platform' => 'xenforo',
            'source' => 'mhf_safeguard_plugin',
            'site_url' => $boardUrl,
            'context' => $context->toPayloadArray(),
            'message' => $cleanMessage,
            'message_hash' => hash('sha256', $cleanMessage),
            'return_spans' => true,
            'return_sentences' => true,
            'sent_at' => time()
        ];
    }
}
