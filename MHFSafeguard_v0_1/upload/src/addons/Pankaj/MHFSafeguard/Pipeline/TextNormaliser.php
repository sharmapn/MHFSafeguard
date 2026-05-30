<?php

namespace Pankaj\MHFSafeguard\Pipeline;

class TextNormaliser
{
    public function normalise(string $message): string
    {
        $message = $this->stripBbCode($message);
        $message = html_entity_decode($message, ENT_QUOTES | ENT_HTML5, 'UTF-8');
        $message = preg_replace('/\r\n|\r/', "\n", $message);
        $message = preg_replace('/[ \t]+/', ' ', $message);
        $message = preg_replace('/\n{3,}/', "\n\n", $message);

        return trim($message);
    }

    protected function stripBbCode(string $message): string
    {
        try
        {
            return \XF::app()->stringFormatter()->stripBbCode($message, [
                'stripQuote' => true,
                'hideUnviewable' => true
            ]);
        }
        catch (\Throwable $e)
        {
            $message = preg_replace('#\[quote[^\]]*\].*?\[/quote\]#is', '', $message);
            $message = preg_replace('#\[[^\]]+\]#', '', $message);
            return $message;
        }
    }
}
